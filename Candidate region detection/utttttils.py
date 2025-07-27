# import numpy as np
# import os
# from collections import defaultdict
#
# import numpy as np
#
#
# def extract_patch(image_data, bbox, patch_size):
#     """
#     从图像数据中提取补丁
#     :param image_data: 输入的图像数据 (numpy array)
#     :param bbox: 边界框坐标，格式为 [置信度, z, y, x, 直径]
#     :param patch_size: 补丁的目标大小 (depth, height, width)
#     :return: 补丁数据 (numpy array)
#     """
#     confidence, z, y, x, diameter = bbox
#
#     # 计算半径（直径的一半），以确定提取区域的大小
#     half_diameter = diameter // 2
#
#     # 计算补丁提取的起始位置和结束位置
#     z_start = max(z - half_diameter, 0)
#     y_start = max(y - half_diameter, 0)
#     x_start = max(x - half_diameter, 0)
#
#     z_end = min(z_start + diameter, image_data.shape[0])
#     y_end = min(y_start + diameter, image_data.shape[1])
#     x_end = min(x_start + diameter, image_data.shape[2])
#
#     # 提取补丁
#     patch = image_data[z_start:z_end, y_start:y_end, x_start:x_end]
#
#     # 如果补丁的尺寸不等于目标尺寸，需要进行调整
#     if patch.shape != patch_size:
#         patch = resize_patch(patch, patch_size)
#
#     return patch
#
#
# def resize_patch(patch, target_size):
#     """
#     调整补丁到目标大小
#     :param patch: 输入补丁 (numpy array)
#     :param target_size: 目标补丁大小 (depth, height, width)
#     :return: 调整后的补丁 (numpy array)
#     """
#     # 如果补丁的尺寸和目标尺寸不同，则进行插值调整
#     from scipy.ndimage import zoom
#
#     # 计算缩放比例
#     zoom_factors = [target_size[0] / patch.shape[0],
#                     target_size[1] / patch.shape[1],
#                     target_size[2] / patch.shape[2]]
#
#     resized_patch = zoom(patch, zoom_factors, order=1)  # 使用线性插值（order=1）
#     return resized_patch
#
#
# def load_raw_data111(config):
#     if os.path.exists(config.output_path) and len(os.listdir(config.output_path)) > 0:
#         print("补丁数据已存在，跳过数据生成步骤。")
#         return
#     else:
#         print("未找到补丁数据，开始生成补丁数据。")
#
#         # 读取从 .npy 文件中加载的预测框数据
#         npy_path = os.path.join(config.data_path, 'predicted_boxes.npy')
#         predicted_boxes = np.load(npy_path)  # 假设你的 .npy 文件内容是这样的形状 (1688454, 5)
#         print(f"Loaded {predicted_boxes.shape[0]} candidate boxes from .npy file.")
#
#         half_size_z = 14
#         half_size_y = 21
#         half_size_x = 21
#         required_size_z = 28
#         required_size_y = 42
#         required_size_x = 42
#
#         os.makedirs(config.output_path, exist_ok=True)
#
#         # 体素坐标数据
#         voxel_coords = predicted_boxes[:, 1:4]  # 获取体素坐标部分 (coordX, coordY, coordZ)
#         confidence_scores = predicted_boxes[:, 0]  # 获取置信度
#         diameters = predicted_boxes[:, 4]  # 获取直径信息
#
#         # 假设我们可以从体素坐标找到对应的三维图像文件（mhd 文件），
#         # 如果你的三维图像文件命名规则与体素坐标关联，需要调整此部分的代码
#         # 比如根据体素坐标的某个部分来查找对应的文件，下面只是一个示例。
#
#         all_files = set(os.listdir(config.data_path))
#         print(f"Total files in directory: {len(all_files)}")
#
#         # 将候选框按文件组织，准备提取每个文件的补丁
#         fn_to_candidates = defaultdict(list)
#         for i, (voxel_coord, score, diameter) in enumerate(zip(voxel_coords, confidence_scores, diameters)):
#             uid = int(voxel_coord[0])  # 假设 UID 是 voxel_coord 的第一个维度
#             fn_to_candidates[uid].append([voxel_coord[0], voxel_coord[1], voxel_coord[2], score, diameter])
#
#         # 遍历每个文件
#         for fn, candidates in fn_to_candidates.items():
#             print(f"Processing file {fn} with {len(candidates)} candidates.")
#
#             filename = os.path.join(config.data_path, f"{fn:03d}.mhd")
#             print(f"Attempting to load file: {filename}")
#
#             if not os.path.exists(filename):
#                 print(f"警告：文件不存在: {filename}")
#                 continue
#
#             try:
#                 image, origin, spacing = load_itk_image(filename)
#             except Exception as e:
#                 print(f"错误：无法加载文件 {filename}，错误信息: {e}")
#                 continue
#
#             whole_brain = normalize_planes(image)
#             print(f"Image shape: {whole_brain.shape}, Origin: {origin}, Spacing: {spacing}")
#
#             patches = []
#
#             # 遍历每个候选框并提取补丁
#             for voxel_coord, score, diameter in candidates:
#                 Z, Y, X = voxel_coord  # 获取体素坐标
#                 print(f"Voxel coordinates: Z={Z}, Y={Y}, X={X}, Score={score}, Diameter={diameter}")
#
#                 # 计算补丁的起始和结束位置
#                 Z_start = int(Z - half_size_z)
#                 Z_end = int(Z + half_size_z)
#                 Y_start = int(Y - half_size_y)
#                 Y_end = int(Y + half_size_y)
#                 X_start = int(X - half_size_x)
#                 X_end = int(X + half_size_x)
#
#                 # 确保不会越界
#                 Z_start = max(Z_start, 0)
#                 Z_end = min(Z_end, whole_brain.shape[0])
#                 Y_start = max(Y_start, 0)
#                 Y_end = min(Y_end, whole_brain.shape[1])
#                 X_start = max(X_start, 0)
#                 X_end = min(X_end, whole_brain.shape[2])
#
#                 # 提取补丁
#                 dat = whole_brain[Z_start:Z_end, Y_start:Y_end, X_start:X_end]
#                 print(f"Extracted patch shape before padding: {dat.shape}")
#
#                 # 对补丁进行填充，确保尺寸为所需大小
#                 pad_z_before = max(0, half_size_z - (Z - Z_start))
#                 pad_z_after = max(0, required_size_z - (Z_end - Z_start) - pad_z_before)
#                 pad_y_before = max(0, half_size_y - (Y - Y_start))
#                 pad_y_after = max(0, required_size_y - (Y_end - Y_start) - pad_y_before)
#                 pad_x_before = max(0, half_size_x - (X - X_start))
#                 pad_x_after = max(0, required_size_x - (X_end - X_start) - pad_x_before)
#
#                 dat = np.pad(dat, ((pad_z_before, pad_z_after),
#                                    (pad_y_before, pad_y_after),
#                                    (pad_x_before, pad_x_after)), mode='constant', constant_values=0)
#                 print(f"Padded dat shape: {dat.shape}")
#
#                 if dat.shape != (required_size_z, required_size_y, required_size_x):
#                     print(f"Padded dat shape is incorrect: {dat.shape}")
#                     continue
#
#                 patches.append(dat)
#
#             if patches:
#                 patches = np.array(patches, dtype=np.float32)
#                 print(f"Saving {patches.shape[0]} patches for file {fn}")
#
#                 # 保存补丁数据
#                 patch_dat_filename = f"{fn}_pbb.npy"
#                 patch_dat_path = os.path.join(config.output_path, patch_dat_filename)
#                 np.save(patch_dat_path, patches)
#                 print(f"Saved patch_dat to {patch_dat_path}")
#             else:
#                 print(f"No valid patches found for file {fn}")
#
#             print(f"Processed {fn}.")
#
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock3D(nn.Module):
    """
    一个简单的 Non-Local Block 3D 实现，用于捕捉更远距离的空间依赖。
    参考文献: Non-local Neural Networks, CVPR 2018
    """

    def __init__(self, in_channels):
        super(NonLocalBlock3D, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // 2)

        # g, θ, φ 都用 1x1x1 卷积降维
        self.g = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)

        # 对输出进行线性映射，使其通道回到原维度
        self.W = nn.Conv3d(self.inter_channels, in_channels, kernel_size=1)
        # 初始化为 0，保证初始时 Non-Local 分支不会破坏原始特征
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        """
        batch_size, C, D, H, W = x.size()

        # g(x) -> [B, C//2, D, H, W] -> [B, C//2, DHW] -> [B, DHW, C//2]
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # [B, DHW, C//2]

        # θ(x) -> [B, C//2, DHW] -> [B, DHW, C//2]
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # φ(x) -> [B, C//2, DHW]
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # 计算相似度 f = θ(x)^T * φ(x)
        f = torch.matmul(theta_x, phi_x)  # [B, DHW, DHW]
        f_div_C = f / f.size(-1)  # 归一化

        # f_div_C * g(x) -> [B, DHW, C//2]
        y = torch.matmul(f_div_C, g_x)
        # 还原为 [B, C//2, D, H, W]
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, D, H, W)

        # 线性映射回原通道数
        y = self.W(y)
        # 残差连接：z = x + y
        z = x + y

        return z


class CMFA(nn.Module):
    def __init__(self, channel, reduction=16, scales=[1, 2], dropout_rate=0.3):
        super(CMFA, self).__init__()
        self.scales = scales
        self.reduction = reduction
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout 防止过拟合

        # 多尺度自适应平均/最大池化层
        self.avg_pools = nn.ModuleList([nn.AdaptiveAvgPool3d((s, s, s)) for s in scales])
        self.max_pools = nn.ModuleList([nn.AdaptiveMaxPool3d((s, s, s)) for s in scales])

        # 空洞卷积层 (dilation=2, padding=2)
        self.dilated_conv = nn.Conv3d(channel, channel, kernel_size=3, dilation=2, padding=2)

        # 用于学习缩放因子的全连接层（示例仍保留，后面会与多尺度融合结果结合）
        self.scale_fc = nn.Linear(channel, len(scales), bias=False)

        # ------------------------- 融合方式的多样化（示例） -------------------------
        # 在拼接多尺度特征前，对每个尺度的特征先做一层独立的线性映射并激活，
        # 再将它们叠加（或拼接）起来进行最终注意力计算。
        self.per_scale_fcs = nn.ModuleList()
        for s in scales:
            # 该尺度下，avg/max_pool 后的特征尺寸 = channel * s * s * s
            in_feats = channel * s * s * s
            self.per_scale_fcs.append(
                nn.Sequential(
                    nn.Linear(in_feats, in_feats, bias=False),
                    nn.BatchNorm1d(in_feats),  # 建议1：增加 BN
                    nn.ReLU(inplace=True)
                )
            )
        # ---------------------------------------------------------------------

        # 全连接层，用于计算最终的通道注意力权重
        # 这里多尺度 avg + max -> 2 * sum(s^3) * channel 个特征
        total_features = 2 * channel * sum([s ** 3 for s in scales])
        self.fc = nn.Sequential(
            nn.Linear(total_features, channel // reduction, bias=False),
            nn.BatchNorm1d(channel // reduction),  # 建议1：增加 BN
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.BatchNorm1d(channel),  # 再加一层 BN
            nn.Sigmoid()
        )

        # 更复杂的空间注意力：在原有卷积 + Sigmoid 前增加一个 NonLocalBlock3D
        self.spatial_attention = nn.Sequential(
            NonLocalBlock3D(channel),  # 建议5：引入 Non-Local
            nn.Conv3d(channel, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 残差连接可以在后面 forward 里进行（建议4）

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        """
        identity = x  # 残差连接使用

        # 先使用空洞卷积，扩大感受野
        x = self.dilated_conv(x)
        print(f"x shape after dilated_conv: {x.shape}")

        b, c, d, h, w = x.size()

        # ----------- 多尺度特征提取 + 多样化的融合方式 -----------
        # avg + max，一共 2 * len(scales) 个特征
        fused_scale_feats = []

        for i, (avg_pool, max_pool, s) in enumerate(zip(self.avg_pools, self.max_pools, self.scales)):
            # 1) 平均池化特征
            avg_feat = avg_pool(x)  # [B, C, s, s, s]
            avg_feat = avg_feat.view(b, c * s * s * s)  # 展平
            # 先做一层线性 + BN + ReLU
            avg_feat = self.per_scale_fcs[i](avg_feat)
            print(f"avg_feat shape at scale {s}: {avg_feat.shape}")

            # 2) 最大池化特征
            max_feat = max_pool(x)  # [B, C, s, s, s]
            max_feat = max_feat.view(b, c * s * s * s)
            max_feat = self.per_scale_fcs[i](max_feat)
            print(f"max_feat shape at scale {s}: {max_feat.shape}")

            # 两者可直接相加，也可拼接，这里拼接示例
            scale_feat = torch.cat([avg_feat, max_feat], dim=1)  # [B, 2*C*s^3]
            print(f"scale_feat shape at scale {s}: {scale_feat.shape}")

            fused_scale_feats.append(scale_feat)

        # 将各尺度特征再次拼接: [B, 2 * sum(C * s^3)]
        ms_feat = torch.cat(fused_scale_feats, dim=1)  # 最终会是 [B, 2*C*(s1^3 + s2^3 + ...)]
        print(f"ms_feat shape: {ms_feat.shape}")

        # ----------- 通过全连接层计算通道注意力 -----------
        channel_attention = self.fc(ms_feat)  # [B, C]
        print(f"channel_attention shape before reshaping: {channel_attention.shape}")
        channel_attention = channel_attention.view(b, c, 1, 1, 1)

        # 使用缩放因子调整（修正这里的代码）
        # 原代码 x.view(b, c) 会导致形状不匹配，改为全局平均池化
        pooled_x = x.mean(dim=[2, 3, 4])  # [B, C]
        print(f"pooled_x shape: {pooled_x.shape}")
        scale_weights = torch.softmax(self.scale_fc(pooled_x), dim=1).view(b, len(self.scales), 1, 1, 1)
        print(f"scale_weights shape: {scale_weights.shape}")
        # 将 scale_weights 的均值乘到 channel_attention 上
        channel_attention = channel_attention * scale_weights.mean(dim=1, keepdim=True)
        print(f"channel_attention shape after scaling: {channel_attention.shape}")

        # Dropout 防止过拟合
        channel_attention = self.dropout(channel_attention)
        print(f"channel_attention shape after dropout: {channel_attention.shape}")

        # ----------- 空间注意力 -----------
        spatial_attention = self.spatial_attention(x)  # [B, 1, D, H, W]
        print(f"spatial_attention shape: {spatial_attention.shape}")

        # ----------- 将通道注意力和空间注意力相乘 -----------
        attention_map = channel_attention * spatial_attention  # [B, C, D, H, W]
        print(f"attention_map shape: {attention_map.shape}")

        # ----------- 残差连接：输出 = x * 注意力 + identity -----------
        out = x * attention_map + identity
        print(f"out shape after residual connection: {out.shape}")

        return out


if __name__ == "__main__":
    # 简单测试
    input_tensor = torch.randn(2, 32, 8, 16, 16)  # batch=2, channel=32, D=8, H=16, W=16
    model = CMFA(channel=32, reduction=8, scales=[1, 2], dropout_rate=0.3)
    output = model(input_tensor)
    print("Input shape :", input_tensor.shape)
    print("Output shape:", output.shape)
