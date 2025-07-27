import torch
from torch import nn

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        # 确保卷积核大小是3或者7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 根据卷积核大小设置填充
        padding = 3 if kernel_size == 7 else 1
        # 定义一个3D卷积层，输入通道数为2（最大值和平均值），输出通道数为1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        # Sigmoid激活函数用于将卷积输出转换为权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x的形状预期为（批量大小，通道数，深度，高度，宽度）
        # 计算每个通道的平均值作为特征图的一个表示
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 计算每个通道的最大值作为另一个特征图的表示
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 将平均值和最大值特征图拼接在一起
        x = torch.cat([avg_out, max_out], dim=1)
        # 通过3D卷积层计算空间注意力
        x = self.conv1(x)
        # 通过Sigmoid激活函数获取注意力权重
        return self.sigmoid(x)

# 示例：在3D CNN块中使用SpatialAttention3D模块
class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3DBlock, self).__init__()
        # 定义一个3D卷积层
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        # 定义批量归一化层
        self.bn = nn.BatchNorm3d(out_channels)
        # 定义ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 实例化空间注意力模块
        self.spatial_attention = SpatialAttention3D()

    def forward(self, x):
        # 通过3D卷积层
        x = self.conv3d(x)
        # 通过批量归一化层
        x = self.bn(x)
        # 通过ReLU激活函数
        x = self.relu(x)
        # 计算空间注意力权重
        attention = self.spatial_attention(x)
        # 将注意力权重应用于原始特征图
        x = x * attention
        return x

# 示例：在3D CNN中添加带空间注意力的Conv3DBlock
class LungNoduleDetectionNet(nn.Module):
    def __init__(self):
        super(LungNoduleDetectionNet, self).__init__()
        # 添加第一层
        self.layer1 = Conv3DBlock(1, 16)
        # 添加第二层
        self.layer2 = Conv3DBlock(16, 32)
        # 根据需要添加更多层

    def forward(self, x):
        # 数据通过第一层
        x = self.layer1(x)
        # 数据通过第二层
        x = self.layer2(x)
        # 继续通过网络
        return x

# 实例化模型
model = LungNoduleDetectionNet()
# 假设'x'是您的输入3D CT扫描数据，形状为（批量大小，通道数，深度，高度，宽度）
output = model(x)