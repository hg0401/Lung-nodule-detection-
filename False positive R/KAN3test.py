import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import os
from scipy.ndimage import zoom
import argparse
from tqdm import tqdm
from multiprocessing import Pool, Manager
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")


# Resampling function
def resample(imgs, spacing, new_spacing, order=1):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs
    else:
        raise ValueError('wrong shape')


# The results of converting world coordinates to voxel coordinates are all output in the order of zyx.
def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.int32(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


# Load the .mhd file
def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing



def lumTrans(img):
    lungwin = np.array([-1200., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


# KANLayer
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, spline_order=2, grid_size=3, use_multiplication=False):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.use_multiplication = use_multiplication

        # Parameter initialization
        self.w_b = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.w_b, mode='fan_in', nonlinearity='relu')  # More stable initialization
        self.w_s = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.w_s, mode='fan_in', nonlinearity='relu')
        self.coefficients = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.01)  # Reduce the initial value
        self.grid = nn.Parameter(torch.linspace(-1.5, 1.5, grid_size + spline_order + 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_features))

        if self.use_multiplication:
            self.multiplication_weights = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_normal_(self.multiplication_weights, mode='fan_in', nonlinearity='relu')

    def bspline_basis(self, x, grid, order):
        """Compute B-spline basis functions with added numerical stability."""
        x = x.unsqueeze(-1)
        grid = grid.unsqueeze(0).unsqueeze(0)
        # Ensure the grid points are ordered.
        grid = torch.sort(grid, dim=-1)[0]
        basis = (x >= grid[..., :-1]) & (x < grid[..., 1:])
        basis = basis.float()
        for k in range(1, order + 1):
            denom_left = grid[..., k:-1] - grid[..., :-k - 1]
            denom_right = grid[..., k + 1:] - grid[..., 1:-k]
            #Prevent division by zero
            denom_left = torch.where(denom_left.abs() < 1e-6, torch.ones_like(denom_left) * 1e-6, denom_left)
            denom_right = torch.where(denom_right.abs() < 1e-6, torch.ones_like(denom_right) * 1e-6, denom_right)
            left = (x - grid[..., :-k - 1]) / denom_left
            right = (grid[..., k + 1:] - x) / denom_right
            basis = left * basis[..., :-1] + right * basis[..., 1:]
        return basis[..., :self.grid_size].clamp(-10, 10)  #Restrict the output range

    def forward(self, x):
        """Forward Propagation"""
        # Input normalization
        x = torch.clamp(x, -10, 10)
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            print("Warning: NaN or Inf in input x")

        is_3d = len(x.shape) > 2
        if is_3d:
            batch_size, channels, D, H, W = x.shape
            x = x.view(batch_size, channels, -1).transpose(1, 2).contiguous()

        # Residual basis function
        silu_x = F.silu(x)
        if torch.any(torch.isnan(silu_x)) or torch.any(torch.isinf(silu_x)):
            print("Warning: NaN or Inf in silu_x")
        if is_3d:
            silu_out = torch.einsum('bpc,oc->bpo', silu_x, self.w_b)
        else:
            silu_out = torch.einsum('bi,oi->bo', silu_x, self.w_b)
        if torch.any(torch.isnan(silu_out)) or torch.any(torch.isinf(silu_out)):
            print("Warning: NaN or Inf in silu_out")

        # split out
        basis = self.bspline_basis(x, self.grid, self.spline_order)
        if torch.any(torch.isnan(basis)) or torch.any(torch.isinf(basis)):
            print("Warning: NaN or Inf in basis")
        if is_3d:
            spline_out = torch.einsum('bpfg,ocg->bpo', basis, self.coefficients * self.w_s.unsqueeze(-1))
        else:
            spline_out = torch.einsum('bfg,ofg->bo', basis, self.coefficients * self.w_s.unsqueeze(-1))
        if torch.any(torch.isnan(spline_out)) or torch.any(torch.isinf(spline_out)):
            print("Warning: NaN or Inf in spline_out")

        # Combined output
        output = silu_out + spline_out

        if self.use_multiplication:
            if is_3d:
                multiplication_out = torch.einsum('bpc,oc->bpo', x, self.multiplication_weights)
            else:
                multiplication_out = torch.einsum('bi,oi->bo', x, self.multiplication_weights)
            output = output * multiplication_out
            if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
                print("Warning: NaN or Inf after multiplication")

        # 添加偏置
        if is_3d:
            output = output + self.bias.view(1, 1, -1)
        else:
            output = output + self.bias.view(1, -1)

        if is_3d:
            output = output.transpose(1, 2).view(batch_size, self.out_features, D, H, W)

        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            print("Warning: NaN or Inf in final output")

        return output.clamp(-10, 10)  # Restrict the final output range

# ConvKAN3D 
class ConvKAN3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.kan_act = KANLayer(out_channels, out_channels)
        self.bn = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.kan_act(x)
        x = self.bn(x)
        x = self.pool(x)
        return x

# SelfKAGNtention3D 
class SelfKAGNtention3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.kan_act = KANLayer(channels, channels)
        self.scale = (channels ** -0.5)

    def forward(self, x):
        B, C, D, H, W = x.shape
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = [self.kan_act(t) for t in qkv]
        q = q.view(B, C, -1).transpose(1, 2)
        k = k.view(B, C, -1).transpose(1, 2)
        v = v.view(B, C, -1).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).view(B, C, D, H, W)
        return x + out


#  KANet3D 
class KANet3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
         
        self.conv1 = ConvKAN3D(1, 16, pool=False)
        self.conv2 = ConvKAN3D(16, 32)
        self.conv3 = ConvKAN3D(32, 64)
        self.conv4 = ConvKAN3D(64, 128)  
        self.conv5 = ConvKAN3D(128, 256)  

        
        self.attn1 = SelfKAGNtention3D(64)
        self.attn2 = SelfKAGNtention3D(256)  
        
        self.up5 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.up4 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.up3 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.up2 = nn.ConvTranspose3d(32, 16, 2, stride=2)

       
        self.fuse_conv = nn.Conv3d(16 + 16 + 32 + 64 + 256, 256, 1)  # Corrected channels: 384 -> 256

       
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.5)  
        self.fc1 = KANLayer(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c3 = self.attn1(c3) 
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c5 = self.attn2(c5)  

        
        up5 = self.up5(c5)
        up4 = self.up4(up5 + c4)  
        up3 = self.up3(up4 + c3)  
        up2 = self.up2(up3 + c2)  

        
        fused = self.fuse_conv(torch.cat([
            c1,  # [batch_size, 16, 32, 32, 32]
            up2,  # [batch_size, 16, 32, 32, 32]
            F.interpolate(up3, scale_factor=2, mode='trilinear'),  # [batch_size, 32, 32, 32, 32]
            F.interpolate(up4, scale_factor=4, mode='trilinear'),  # [batch_size, 64, 32, 32, 32]
            F.interpolate(c5, size=c1.shape[2:], mode='trilinear')  # [batch_size, 256, 32, 32, 32]
        ], dim=1))

        out = self.pool(fused).view(x.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def get_kanet3d(num_classes=2):
    return KANet3D(num_classes=num_classes)



#Process all nodules for a single seriesuid.
def process_series(args):
    seriesuid, group, luna_data, roi_size, resolution, shared_dict = args
    seriesuid = int(seriesuid)

    # Load the original image
    img_path = os.path.join(luna_data, f'{seriesuid:03d}.mhd')
    if not os.path.exists(img_path):
        print(f"Image {img_path} not found, skipping.")
        return None

    # Check whether it has been cached
    if seriesuid in shared_dict:
        sliceim, origin, spacing = shared_dict[seriesuid]
    else:
        sliceim, origin, spacing = load_itk_image(img_path)
        sliceim = lumTrans(sliceim)
        sliceim = resample(sliceim, spacing, resolution)
        shared_dict[seriesuid] = (sliceim, origin, spacing)

    rois = []
    infos = []

    for _, row in group.iterrows():
        coordX = float(row['coordX'])
        coordY = float(row['coordY'])
        coordZ = float(row['coordZ'])
        diameter_mm = float(row['diameter_mm'])
        # diameter_mm = float(row['threshold'])
        det_prob = float(row['probability'])

        #  Convert world coordinates to voxel coordinates
        worldCoord = np.array([coordZ, coordY, coordX])
        voxelCoord = worldToVoxelCoord(worldCoord, origin, spacing)
        # print("voxelCoord",voxelCoord)
        # print(spacing,resolution)
        voxelCoord_resampled = voxelCoord * spacing / resolution
        # print("voxelCoord_resampled",voxelCoord_resampled)

        # Extract the ROI
        center = voxelCoord_resampled.astype(int)
        half_size = roi_size // 2
        z_start = max(0, center[0] - half_size)
        z_end = min(sliceim.shape[0], center[0] + half_size)
        y_start = max(0, center[1] - half_size)
        y_end = min(sliceim.shape[1], center[1] + half_size)
        x_start = max(0, center[2] - half_size)
        x_end = min(sliceim.shape[2], center[2] + half_size)

        # print(z_start,y_start,x_start)
        roi = sliceim[z_start:z_end, y_start:y_end, x_start:x_end]

        # Fill the ROI
        if roi.shape[0] < roi_size or roi.shape[1] < roi_size or roi.shape[2] < roi_size:
            pad_z = roi_size - roi.shape[0]
            pad_y = roi_size - roi.shape[1]
            pad_x = roi_size - roi.shape[2]
            roi = np.pad(roi, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=170)

        #Normalization
        roi = (roi - 128) / 128
        roi = roi[np.newaxis, ...]

        rois.append(roi)
        infos.append((seriesuid, coordX, coordY, coordZ, diameter_mm, det_prob))

    return rois, infos


# main preprocessing and prediction functions
def preprocess_and_predict(csv_path, luna_data, model_path, output_csv, roi_size=32, batch_size=32, num_workers=4):
    # load CSV file
    df = pd.read_csv(csv_path)
    resolution = np.array([1, 1, 1])

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_kanet3d(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Group Processing
    grouped = df.groupby('seriesuid')
    manager = Manager()
    shared_dict = manager.dict()  # Shared Cache

    # Multiprocessing
    args = [(seriesuid, group, luna_data, roi_size, resolution, shared_dict) for seriesuid, group in grouped]
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_series, args)

    # Collect all ROIs and associated
    all_rois = []
    all_infos = []
    for result in results:
        if result is not None:
            rois, infos = result
            all_rois.extend(rois)
            all_infos.extend(infos)

    # Batch Prediction
    results = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_rois), batch_size), desc="Predicting"):
            batch_rois = all_rois[i:i + batch_size]
            batch_infos = all_infos[i:i + batch_size]
            batch_rois = np.stack(batch_rois)
            inputs = torch.tensor(batch_rois, dtype=torch.float32).to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            for j, (suid, x, y, z, diam, det_p) in enumerate(batch_infos):
                pred_prob = probs[j]
                # if (pred_prob*0.3+det_p*0.7)>0.57:
                # if (pred_prob * 0.1 + det_p * 0.9) > 0.666:
                # if det_p > 0.688:
                # if det_p > 0.5:
                if pred_prob > 0.2:
                # 632
                    results.append({
                        'seriesuid': suid,
                        'coordX': x,
                        'coordY': y,
                        'coordZ': z,
                        'diameter_mm': diam,
                        'probability': det_p,
                        'classification_probability': pred_prob,
                        # 'prediction': int(pred_prob > 0.7)
                    })


    # save result
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    # print(f"Results saved to {output_csv}")
    print(f"Results saved to {output_csv} (only predictions = 1, total: {len(results)} rows)")

def main():
    parser = argparse.ArgumentParser(description="Lung Nodule Classification Prediction")
    parser.add_argument('--csv_path', default="D:\\HG\\3DCNN\\classification\\outputcsv\\predanno0.3-0.csv", type=str,help='Path to input CSV file')
    parser.add_argument('--luna_data', default='D:/HG/3DCNN/DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection-master/Luna_data/', type=str,help='Path to LUNA data directory')
    parser.add_argument('--model_path', default='D:\\HG\\3DCNN\\classification\\models\\best_model.pth',type=str,help='Path to trained model')
   # parser.add_argument('--model_path', default='D:\\HG\\3DCNN\\classification\\KAN3model\\KAN3deep\\best_model.pth',type=str, help='Path to trained model')
    parser.add_argument('--output_csv', default='D:\\HG\\3DCNN\\classification\\outputcsv\\0428.csv', type=str,help='Path to output CSV file')
    parser.add_argument('--roi_size', default=32, type=int, help='ROI size')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for prediction')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of worker processes')

    args = parser.parse_args()

    preprocess_and_predict(
        csv_path=args.csv_path,
        luna_data=args.luna_data,
        model_path=args.model_path,
        output_csv=args.output_csv,
        roi_size=args.roi_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()