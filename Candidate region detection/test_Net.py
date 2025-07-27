



from layers_se import *

# 如果 Multi-Scale CMFA3D 也在本文件，可以直接引用
# 若在 layers_se.py 中，则 from layers_se import CMFA3D_MultiScale
# 这里假设已经写在同一文件中：
# from layers_se import CMFA3D_MultiScale

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#
# class ChannelAttention3D(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(ChannelAttention3D, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.max_pool = nn.AdaptiveMaxPool3d(1)
#         self.conv_down = nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False)
#         self.conv_up = nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.conv_up(self.relu(self.conv_down(self.avg_pool(x))))
#         max_out = self.conv_up(self.relu(self.conv_down(self.max_pool(x))))
#         ca_weight = self.sigmoid(avg_out + max_out)  # (B, C, 1, 1, 1)
#         return x * ca_weight
#
#
# class SpatialAttention3D_MultiScale(nn.Module):
#     """
#     多尺度空间注意力：
#       - 三个分支：
#         1. 一个 kernel_size=3 的卷积层
#         2. 两个 kernel_size=3 的卷积层
#         3. 三个 kernel_size=3 的卷积层
#       - 每个分支的最后一个卷积层将输出映射到1个通道
#       - 最终通过1x1卷积融合所有分支的输出，生成统一的空间注意力权重图
#     """
#
#     def __init__(self, num_layers_per_branch=[1, 2,3], kernel_size=3, mid_channels=8):
#         super(SpatialAttention3D_MultiScale, self).__init__()
#         self.branches = nn.ModuleList()
#         for i, num_layers in enumerate(num_layers_per_branch):
#             layers = []
#             in_channels = 2  # 输入是max + avg池化拼接后的2个通道
#             for layer_idx in range(num_layers):
#                 layers.append(nn.Conv3d(
#                     in_channels=in_channels,
#                     out_channels=mid_channels,
#                     kernel_size=kernel_size,
#                     padding=kernel_size // 2,
#                     bias=False
#                 ))
#                 layers.append(nn.ReLU(inplace=True))
#                 in_channels = mid_channels
#             # 最后一层卷积映射到1个通道
#             layers.append(nn.Conv3d(
#                 in_channels=mid_channels,
#                 out_channels=1,
#                 kernel_size=1,
#                 padding=0,
#                 bias=False
#             ))
#             self.branches.append(nn.Sequential(*layers))
#
#         # 分支融合：将所有分支的输出拼接，然后通过1x1卷积融合为1个通道
#         self.fuse_conv = nn.Conv3d(len(num_layers_per_branch), 1, kernel_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         """
#         x: (B, 2, D, H, W)  # max + avg池化后的拼接
#         """
#         branch_outputs = []
#         for branch in self.branches:
#             branch_out = branch(x)  # (B,1,D,H,W)
#             branch_outputs.append(branch_out)
#
#         # 拼接分支输出 (B, num_branches, D, H, W)
#         multi_scale_feat = torch.cat(branch_outputs, dim=1)  # (B, 3, D, H, W)
#         fused = self.fuse_conv(multi_scale_feat)             # (B,1,D,H,W)
#         sa_weight = self.sigmoid(fused)                      # (B,1,D,H,W)
#
#         return sa_weight
#
#
# class CMFA3D_MultiScale(nn.Module):
#     """
#     通道注意力 + 多尺度空间注意力
#     """
#     def __init__(self, in_channels, reduction=16,num_layers_per_branch=[1, 2,3], kernel_size=3, mid_channels=8):
#         super(CMFA3D_MultiScale, self).__init__()
#         self.channel_attention = ChannelAttention3D(in_channels, reduction)
#         self.spatial_attention = SpatialAttention3D_MultiScale(
#             num_layers_per_branch=num_layers_per_branch,
#             kernel_size=kernel_size,
#             mid_channels=mid_channels
#         )
#
#     def forward(self, x):
#         # 通道注意力
#         x_ca = self.channel_attention(x)  # (B, C, D, H, W)
#
#         # 空间注意力：先做 max + avg 池化并拼接
#         pool_max, _ = torch.max(x_ca, dim=1, keepdim=True)  # (B,1,D,H,W)
#         pool_avg = torch.mean(x_ca, dim=1, keepdim=True)    # (B,1,D,H,W)
#         pool_cat = torch.cat([pool_max, pool_avg], dim=1)   # (B,2,D,H,W)
#
#         # 空间注意力权重
#         sa_weight = self.spatial_attention(pool_cat)        # (B,1,D,H,W)
#
#         # 最终输出
#         out = x_ca * sa_weight
#         return out


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
        # 通道注意力
        avg_out = self.conv_up(self.relu(self.conv_down(self.avg_pool(x))))
        max_out = self.conv_up(self.relu(self.conv_down(self.max_pool(x))))
        ca_weight = self.sigmoid(avg_out + max_out)  # (B, C, 1, 1, 1)

        # 残差连接
        return x + x * ca_weight  # 残差连接


# class DynamicSpatialAttention3D(nn.Module):
#     def __init__(self, num_layers_range=(1, 3), kernel_size=3, mid_channels=8):
#         super(DynamicSpatialAttention3D, self).__init__()
#         self.num_layers_range = num_layers_range
#         self.kernel_size = kernel_size
#         self.mid_channels = mid_channels
#
#         # 动态分支的卷积层
#         self.branches = nn.ModuleList()
#         for num_layers in range(num_layers_range[0], num_layers_range[1] + 1):
#             layers = []
#             in_channels = 2  # 输入是 max + avg 池化拼接后的2个通道
#             for _ in range(num_layers):
#                 layers.append(nn.Conv3d(
#                     in_channels=in_channels,
#                     out_channels=mid_channels,
#                     kernel_size=kernel_size,
#                     padding=kernel_size // 2,
#                     bias=False
#                 ))
#                 layers.append(nn.ReLU(inplace=True))
#                 in_channels = mid_channels
#             # 最后一层卷积映射到1个通道
#             layers.append(nn.Conv3d(
#                 in_channels=mid_channels,
#                 out_channels=1,
#                 kernel_size=1,
#                 padding=0,
#                 bias=False
#             ))
#             self.branches.append(nn.Sequential(*layers))
#
#         # 分支融合：将所有分支的输出拼接，然后通过1x1卷积融合为1个通道
#         self.fuse_conv = nn.Conv3d(len(self.branches), 1, kernel_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         """
#         x: (B, 2, D, H, W)  # max + avg 池化后的拼接
#         """
#         # 根据输入分辨率动态选择分支
#         resolution = x.size()[2:]  # 获取空间分辨率 (D, H, W)
#         if resolution[0] * resolution[1] * resolution[2] < 32**3:  # 小分辨率
#             selected_branch_idx = 0  # 选择较浅的分支
#         else:  # 大分辨率
#             selected_branch_idx = len(self.branches) - 1  # 选择较深的分支
#
#         # 使用选定的分支计算空间注意力权重
#         sa_weight = self.branches[selected_branch_idx](x)  # (B, 1, D, H, W)
#         sa_weight = self.sigmoid(sa_weight)
#
#         return sa_weight



class DynamicSpatialAttention3D(nn.Module):
    def __init__(self, num_layers_range=(1, 3), kernel_size=3, mid_channels=8):
        super(DynamicSpatialAttention3D, self).__init__()
        self.num_layers_range = num_layers_range
        self.kernel_size = kernel_size
        self.mid_channels = mid_channels

        # 动态分支的卷积层
        self.branches = nn.ModuleList()
        for num_layers in range(num_layers_range[0], num_layers_range[1] + 1):
            layers = []
            in_channels = 2  # 输入是 max + avg 池化拼接后的2个通道
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
            # 最后一层卷积映射到1个通道
            layers.append(nn.Conv3d(
                in_channels=mid_channels,
                out_channels=1,
                kernel_size=1,
                padding=0,
                bias=False
            ))
            self.branches.append(nn.Sequential(*layers))

        # 分支融合：将所有分支的输出拼接，然后通过1x1卷积融合为1个通道
        self.fuse_conv = nn.Conv3d(len(self.branches), 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, 2, D, H, W)  # max + avg 池化后的拼接
        """
        # 计算所有分支的输出
        branch_outputs = [branch(x) for branch in self.branches]  # 每个分支的输出 (B, 1, D, H, W)

        # 计算输入分辨率
        resolution = x.size()[2:]  # 获取空间分辨率 (D, H, W)
        resolution_prod = resolution[0] * resolution[1] * resolution[2]

        # 使用 PyTorch 操作替代 Python 控制流
        # 创建动态权重：小分辨率时权重为 [1, 0, 0]，大分辨率时权重为 [0, 0, 1]
        weights = torch.zeros(len(self.branches), device=x.device)  # 初始化权重
        weights[0] = (resolution_prod < 32**3).float()  # 小分辨率时 weights[0] = 1
        weights[-1] = (resolution_prod >= 32**3).float()  # 大分辨率时 weights[-1] = 1

        # 加权融合
        weighted_outputs = torch.stack(branch_outputs, dim=0)  # (num_branches, B, 1, D, H, W)
        weights = weights.view(-1, 1, 1, 1, 1)  # 调整权重形状以匹配分支输出
        fused = torch.sum(weighted_outputs * weights, dim=0)  # (B, 1, D, H, W)
        sa_weight = self.sigmoid(fused)

        return sa_weight


class CMFA3D_MultiScale(nn.Module):
    """
    通道注意力 + 多尺度空间注意力
    """

    def __init__(self, in_channels, reduction=16, num_layers_per_branch=(1,3), kernel_size=3, mid_channels=8):
        super(CMFA3D_MultiScale, self).__init__()
        self.channel_attention = ChannelAttention3D(in_channels, reduction)
        self.spatial_attention = DynamicSpatialAttention3D(
            num_layers_range=num_layers_per_branch,
            kernel_size=kernel_size,
            mid_channels=mid_channels
        )

    def forward(self, x):
        # 通道注意力 + 残差连接
        x_ca = self.channel_attention(x)  # (B, C, D, H, W)

        # 空间注意力：先做 max + avg 池化并拼接
        pool_max, _ = torch.max(x_ca, dim=1, keepdim=True)  # (B,1,D,H,W)
        pool_avg = torch.mean(x_ca, dim=1, keepdim=True)  # (B,1,D,H,W)
        pool_cat = torch.cat([pool_max, pool_avg], dim=1)  # (B,2,D,H,W)

        # 空间注意力权重
        sa_weight = self.spatial_attention(pool_cat)  # (B,1,D,H,W)

        # 空间注意力 + 残差连接
        out = x_ca + x_ca * sa_weight  # 残差连接
        return out

class DoubleConv(nn.Module):
    """(卷积 => [BN] => ReLU) * 2"""
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
    """下采样 => DoubleConv"""
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
    典型单 skip 的 3D U-Net 上采样模块
    1) 上采样：可选转置卷积或三线性插值
    2) 拼接 skip
    3) DoubleConv
    """
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        self.bilinear = bilinear

        if bilinear:
            # 三线性插值
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            # 转置卷积
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x1, x2):
        # 上采样
        x1 = self.up(x1)

        # 若尺寸不匹配，这里可进行 pad 或 interpolate
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]
        x1 = F.pad(
            x1,
            [diffW // 2, diffW - diffW // 2,
             diffH // 2, diffH - diffH // 2,
             diffD // 2, diffD - diffD // 2]
        )

        # 拼接
        x = torch.cat([x2, x1], dim=1)

        # 双卷积
        x = self.conv(x)
        return x


class Net(nn.Module):
    def __init__(self, config, use_attention1=True,use_attention2=True):
        super(Net, self).__init__()
        self.config = config
        self.num_anchors = len(self.config['anchors'])

        # 编码器
        self.inc = DoubleConv(1, 24)
        self.down1 = Down(24, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)

        # 解码器
        self.up1 = Up(256, 128, 128, bilinear=False)
        self.up2 = Up(128, 64, 64, bilinear=False)
        self.up3 = Up(64, 32, 32, bilinear=False)

        self.down5 = nn.Conv3d(32, 32, kernel_size=2, stride=2)

        # 输出头
        # 假设 coord 有 3 个通道
        output_channels = 32 + 3  # 32 + coord(3) = 35

        self.nodule_output = nn.Sequential(
            nn.Conv3d(output_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(64, self.num_anchors, kernel_size=1)
        )
        self.regress_output = nn.Sequential(
            nn.Conv3d(output_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(64, 4 * self.num_anchors, kernel_size=1)
        )

        # 初始化 focal bias
        focal_bias = -math.log((1.0 - 0.01) / 0.01)
        self.nodule_output[2].bias.data.fill_(focal_bias)

        # -----------------------
        # 注意力模块
        # -----------------------
        if use_attention1:
            # 编码器各层注意力
            self.att_x0 = CMFA3D_MultiScale(24, 16, [1,3,5], 3, 8)
            self.att_x1 = CMFA3D_MultiScale(32, 16, [1,3,5], 3, 8)
            self.att_x2 = CMFA3D_MultiScale(64, 16, [1,3,5], 3, 8)
            self.att_x3 = CMFA3D_MultiScale(128,16, [1,3,5], 3, 8)
            self.att_x4 = CMFA3D_MultiScale(256,16, [1,3,5], 3, 8)


        else:
            # 不使用注意力时，Identity 直接返回输入
            self.att_x0 = nn.Identity()
            self.att_x1 = nn.Identity()
            self.att_x2 = nn.Identity()
            self.att_x3 = nn.Identity()
            self.att_x4 = nn.Identity()

        if use_attention2:
            # 解码器各层注意力
            self.att_up1 = CMFA3D_MultiScale(128, 16, [1, 3, 5], 3, 8)
            self.att_up2 = CMFA3D_MultiScale(64,  16, [1, 3, 5], 3, 8)
            self.att_up3 = CMFA3D_MultiScale(32,  16, [1, 3, 5], 3, 8)
        else:
            self.att_up1 = nn.Identity()
            self.att_up2 = nn.Identity()
            self.att_up3 = nn.Identity()

    def forward(self, x, coord):
        # -----------------------
        # 编码器
        # -----------------------
        x0 = self.inc(x)
        x0 = self.att_x0(x0)

        x1 = self.down1(x0)
        x1 = self.att_x1(x1)

        x2 = self.down2(x1)
        x2 = self.att_x2(x2)

        x3 = self.down3(x2)
        x3 = self.att_x3(x3)

        x4 = self.down4(x3)
        x4 = self.att_x4(x4)

        # -----------------------
        # 解码器
        # -----------------------
        x = self.up1(x4, x3)
        x = self.att_up1(x)

        x = self.up2(x, x2)
        x = self.att_up2(x)

        x = self.up3(x, x1)
        x = self.att_up3(x)

        x = self.down5(x)  # => (B,32,D/4,H/4,W/4)

        # 插入坐标信息
        coord_upsampled = F.interpolate(coord, size=x.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, coord_upsampled], dim=1)  # => (B, 35, D/4,H/4,W/4)

        # 输出分支
        nodule_out = self.nodule_output(x)      # => (B,num_anchors,D/4,H/4,W/4)
        regress_out = self.regress_output(x)    # => (B,4*num_anchors,D/4,H/4,W/4)

        # 调整输出形状
        nodule_out = nodule_out.permute(0, 2, 3, 4, 1).contiguous()   # (B,D/4,H/4,W/4,num_anchors)
        regress_out = regress_out.permute(0, 2, 3, 4, 1).contiguous() # (B,D/4,H/4,W/4,4*num_anchors)

        regress_out = regress_out.view(
            regress_out.size(0),
            regress_out.size(1),
            regress_out.size(2),
            regress_out.size(3),
            self.num_anchors,
            4
        )  # => (B,D/4,H/4,W/4,num_anchors,4)

        # 最终合并输出 (sigmoid 分类 + 回归)
        out = torch.cat((nodule_out.unsqueeze(-1), regress_out), dim=5)  # (B,D/4,H/4,W/4,num_anchors,5)
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

    net = Net(config,use_attention1=True,use_attention2=False)
    # net.apply(weights_init)

    # 您需要根据自己的项目定义的 loss 和 GetPBB，这里暂时假设为原来的逻辑
    loss = FocalLoss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb
