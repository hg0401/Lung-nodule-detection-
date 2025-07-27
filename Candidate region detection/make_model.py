import torch
import torch.nn as nn
from res18_se import Net# 导入你的模型定义
# from test_Net import Net# 导入你的模型定义

config = {}
config['anchors'] = [5.0, 10.0, 20.0]
config['chanel'] = 1
config['crop_size'] = [128, 128, 128]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 3.0  # mm, smallest nodule size
config['sizelim2'] = 10
config['sizelim3'] = 20
config['aug_scale'] = True
config['r_rand_crop'] = 0.5
config['pad_value'] = 0
config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38', '990fbe3f0a1b53878669967b9afd1441',
                       'adc3bbc63d40f8761c59be10f1e504c3']

# 创建模型实例
model = Net(config,use_attention1=True,use_attention2=True)  # 用你的模型类替换 YourModel

# 确保模型处于评估模式
model.eval()
# 指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 准备输入数据
input_tensor = torch.randn(1, 1, 128, 128, 128).to(device)
coord_tensor = torch.randn(1, 3, 32, 32, 32).to(device)  # 假设 coord_tensor 与 input_tensor 尺寸相同
# 指定 ONNX 文件保存路径
onnx_file_path = "U-Net0110_aRes_CMFA12_MC135123.onnx"
# 导出模型
torch.onnx.export(model,
                 (input_tensor, coord_tensor),  # 将两个张量作为元组传递
                 onnx_file_path,
                 export_params=True,
                 opset_version=11,
                 do_constant_folding=True,
                 input_names=['input', 'coord'],  # 更新输入张量的名称列表
                 output_names=['output'],
                 dynamic_axes={'input': {0: 'batch_size'},
                               'coord': {0: 'batch_size'},
                               'output': {0: 'batch_size'}})
print(f"Model has been saved to {onnx_file_path}")




#计算模型参数量
total_params = sum(p.numel() for p in model.parameters())

print(f"Total number of parameters: {total_params}")