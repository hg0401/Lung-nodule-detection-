from layers_se import *
from torch.nn import Parameter
import numpy as np
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.conv_down = nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False)
        self.conv_up = nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.conv_up(self.relu(self.conv_down(self.avg_pool(x))))
        max_out = self.conv_up(self.relu(self.conv_down(self.max_pool(x))))
        ca_weight = self.sigmoid(avg_out + max_out)  # (B, C, 1, 1, 1)

        # Residual connection
        return x + x * ca_weight


class SpatialAttention3D_MultiScale(nn.Module):
    def __init__(self, num_layers_range=(1, 3), kernel_size=3, mid_channels=8):
        super(SpatialAttention3D_MultiScale, self).__init__()
        self.num_layers_range = num_layers_range
        self.kernel_size = kernel_size
        self.mid_channels = mid_channels

        # Convolution layers for each dynamic branch
        self.branches = nn.ModuleList()
        for num_layers in range(num_layers_range[0], num_layers_range[1] + 1):
            layers = []
            in_channels = 2  # max + avg pooled concatenated channels
            for _ in range(num_layers):
                layers.append(nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False
                ))
                layers.append(nn.ReLU(inplace=True))
                in_channels = mid_channels
            # Final conv layer maps to 1 channel
            layers.append(nn.Conv3d(
                in_channels=mid_channels,
                out_channels=1,
                kernel_size=1,
                padding=0,
                bias=False
            ))
            self.branches.append(nn.Sequential(*layers))

        # Fuse outputs from all branches using 1x1 conv
        self.fuse_conv = nn.Conv3d(len(self.branches), 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, 2, D, H, W)  # concatenated max + avg pooling
        """
        # Compute outputs from all branches
        branch_outputs = [branch(x) for branch in self.branches]  # (B, 1, D, H, W)

        # Compute input resolution
        resolution = x.size()[2:]  # (D, H, W)
        resolution_prod = resolution[0] * resolution[1] * resolution[2]

        # Dynamic weighting based on resolution
        if resolution_prod < 32**3:
            weights = torch.tensor([0.6, 0.2, 0.2], device=x.device)
        else:
            weights = torch.tensor([0.2, 0.2, 0.6], device=x.device)

        # Weighted fusion
        weighted_outputs = [output * weight for output, weight in zip(branch_outputs, weights)]
        fused = torch.sum(torch.stack(weighted_outputs), dim=0)  # (B, 1, D, H, W)
        sa_weight = self.sigmoid(fused)

        return sa_weight


class CMFA3D_MultiScale(nn.Module):
    """
    Channel attention + Multi-scale spatial attention
    """
    def __init__(self, in_channels, reduction=16, num_layers_per_branch=(1,3), kernel_size=3, mid_channels=8):
        super(CMFA3D_MultiScale, self).__init__()
        self.channel_attention = ChannelAttention3D(in_channels, reduction)
        self.spatial_attention = SpatialAttention3D_MultiScale(
            num_layers_range=num_layers_per_branch,
            kernel_size=kernel_size,
            mid_channels=mid_channels
        )

    def forward(self, x):
        x_origin = x

        # Channel attention + residual
        x_ca = self.channel_attention(x)

        # Spatial attention: max + avg pooling concatenation
        pool_max, _ = torch.max(x_ca, dim=1, keepdim=True)
        pool_avg = torch.mean(x_ca, dim=1, keepdim=True)
        pool_cat = torch.cat([pool_max, pool_avg], dim=1)

        # Spatial attention weight
        sa_weight = self.spatial_attention(pool_cat)

        # Spatial attention + residual
        out = x_ca + x_ca * sa_weight
        out = out + x_origin
        return out


class DoubleConv(nn.Module):
    """(conv => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling => DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Typical 3D U-Net upsampling with single skip connection
    1) Upsample via transposed conv or trilinear interpolation
    2) Concatenate skip
    3) DoubleConv
    """
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatch by padding
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]
        x1 = F.pad(
            x1,
            [diffW // 2, diffW - diffW // 2,
             diffH // 2, diffH - diffH // 2,
             diffD // 2, diffD - diffD // 2]
        )

        # Concatenate skip
        x = torch.cat([x2, x1], dim=1)

        # Double conv
        x = self.conv(x)
        return x


class Net(nn.Module):
    def __init__(self, config, use_attention1=True, use_attention2=True):
        super(Net, self).__init__()
        self.config = config
        self.num_anchors = len(self.config['anchors'])

        # Encoder
        self.inc = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(131, 256)
        self.down4 = Down(256, 512)

        # Decoder
        self.up1 = Up(512, 256, 256, bilinear=False)
        self.up2 = Up(256, 128, 128, bilinear=False)
        self.up3 = Up(128, 64, 64, bilinear=False)

        self.down5 = nn.Conv3d(64, 64, kernel_size=2, stride=2)

        # Output heads
        output_channels = 64

        self.nodule_output = nn.Sequential(
            nn.Conv3d(output_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(128, self.num_anchors, kernel_size=1)
        )
        self.regress_output = nn.Sequential(
            nn.Conv3d(output_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(128, 4 * self.num_anchors, kernel_size=1)
        )

        # Focal bias initialization
        focal_bias = -math.log((1.0 - 0.01) / 0.01)
        self.nodule_output[2].bias.data.fill_(focal_bias)

        # Attention modules
        if use_attention1:
            self.att_x0 = CMFA3D_MultiScale(32, 16, [1, 3, 5], 3, 8)
            self.att_x1 = CMFA3D_MultiScale(64, 16, [1, 3, 5], 3, 8)
            self.att_x2 = CMFA3D_MultiScale(128, 16, [1, 2, 3], 3, 8)
            self.att_x3 = CMFA3D_MultiScale(256, 16, [1, 2, 3], 3, 8)
            self.att_x4 = CMFA3D_MultiScale(512, 16, [1, 2, 3], 3, 8)
        else:
            self.att_x0 = nn.Identity()
            self.att_x1 = nn.Identity()
            self.att_x2 = nn.Identity()
            self.att_x3 = nn.Identity()
            self.att_x4 = nn.Identity()

        if use_attention2:
            self.att_up1 = CMFA3D_MultiScale(256, 16, [1, 2, 3], 3, 8)
            self.att_up2 = CMFA3D_MultiScale(128, 16, [1, 2, 3], 3, 8)
            self.att_up3 = CMFA3D_MultiScale(64, 16, [1, 3, 5], 3, 8)
        else:
            self.att_up1 = nn.Identity()
            self.att_up2 = nn.Identity()
            self.att_up3 = nn.Identity()

    def forward(self, x, coord):
        # Encoder
        x0 = self.inc(x)
        x0 = self.att_x0(x0)

        x1 = self.down1(x0)
        x1 = self.att_x1(x1)

        x2 = self.down2(x1)
        x2 = self.att_x2(x2)

        x2 = torch.cat([x2, coord], dim=1)

        x3 = self.down3(x2)
        x3 = self.att_x3(x3)

        x4 = self.down4(x3)
        x4 = self.att_x4(x4)

        # Decoder
        x = self.up1(x4, x3)
        x = self.att_up1(x)

        x = self.up2(x, x2[:, :128, :, :, :])
        x = self.att_up2(x)

        x = self.up3(x, x1)
        x = self.att_up3(x)

        x = self.down5(x)

        nodule_out = self.nodule_output(x)
        regress_out = self.regress_output(x)

        nodule_out = nodule_out.permute(0, 2, 3, 4, 1).contiguous()
        regress_out = regress_out.permute(0, 2, 3, 4, 1).contiguous()

        regress_out = regress_out.view(
            regress_out.size(0),
            regress_out.size(1),
            regress_out.size(2),
            regress_out.size(3),
            self.num_anchors,
            4
        )

        out = torch.cat((nodule_out.unsqueeze(-1), regress_out), dim=5)
        return out


def get_model():
    config = {}
    config['anchors'] = [5.0, 10.0, 20.0]
    config['chanel'] = 1
    config['crop_size'] = [128, 128, 128]
    config['stride'] = 4
    config['max_stride'] = 16
    config['num_neg'] = 1000
    config['th_neg'] = 0.02
    config['th_pos_train'] = 0.5
    config['th_pos_val'] = 1
    config['num_hard'] = 2
    config['bound_size'] = 12
    config['reso'] = 1
    config['sizelim'] = 3.0
    config['sizelim2'] = 10
    config['sizelim3'] = 20
    config['aug_scale'] = True
    config['r_rand_crop'] = 0.5
    config['pad_value'] = 0
    config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
    config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38', '990fbe3f0a1b53878669967b9afd1441',
                           'adc3bbc63d40f8761c59be10f1e504c3']

    net = Net(config, use_attention1=True, use_attention2=True)
    # net.apply(weights_init)

    loss = FocalLoss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb
