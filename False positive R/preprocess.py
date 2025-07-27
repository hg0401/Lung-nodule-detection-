import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
from scipy.ndimage import zoom, binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def resample(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing
    else:
        raise ValueError('wrong shape')


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


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


def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 1.5 * np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask


def process_series(args):
    seriesuid, group, luna_data, luna_segment, savepath, roi_size, resolution = args
    seriesuid = int(seriesuid)

    
    img_path = os.path.join(luna_data, f'{seriesuid:03d}.mhd')
    if not os.path.exists(img_path):
        print(f"Image {img_path} not found, skipping.")
        return None

    sliceim, origin, spacing = load_itk_image(img_path)


    mask_path = os.path.join(luna_segment, f'{seriesuid:03d}.mhd')
    if os.path.exists(mask_path):
        Mask, _, _ = load_itk_image(mask_path)
        m1 = Mask == 3
        m2 = Mask == 4
        Mask = m1 + m2
        dilatedMask = process_mask(Mask)
    else:
        dilatedMask = np.ones_like(sliceim)

    
    sliceim = lumTrans(sliceim)
    sliceim = sliceim * dilatedMask + 170 * (1 - dilatedMask).astype('uint8')

   
    sliceim, _ = resample(sliceim, spacing, resolution, order=1)

    
    rois = []
    labels = []

    for idx, row in group.iterrows():
        coordX = float(row['coordX'])
        coordY = float(row['coordY'])
        coordZ = float(row['coordZ'])
        class_label = int(row['class'])

        
        worldCoord = np.array([coordZ, coordY, coordX])
        voxelCoord = worldToVoxelCoord(worldCoord, origin, spacing)

        
        voxelCoord_resampled = voxelCoord * spacing / resolution

        
        center = voxelCoord_resampled.astype(int)
        half_size = roi_size // 2
        z_start = max(0, center[0] - half_size)
        z_end = min(sliceim.shape[0], center[0] + half_size)
        y_start = max(0, center[1] - half_size)
        y_end = min(sliceim.shape[1], center[1] + half_size)
        x_start = max(0, center[2] - half_size)
        x_end = min(sliceim.shape[2], center[2] + half_size)

        roi = sliceim[z_start:z_end, y_start:y_end, x_start:x_end]

      
        if roi.shape[0] < roi_size or roi.shape[1] < roi_size or roi.shape[2] < roi_size:
            pad_z = roi_size - roi.shape[0]
            pad_y = roi_size - roi.shape[1]
            pad_x = roi_size - roi.shape[2]
            roi = np.pad(roi, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=170)

        rois.append(roi)
        labels.append(class_label)

  
    np.savez(os.path.join(savepath, f'{seriesuid:03d}_data.npz'),
             rois=np.array(rois), labels=np.array(labels))
    print(f"Processed seriesuid {seriesuid} with {len(group)} nodules")


def preprocess_for_classification(csv_path, luna_data, luna_segment, savepath, roi_size=32, num_workers=4):
    annos = pd.read_csv(csv_path)
    resolution = np.array([1, 1, 1]) 

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    
    grouped = annos.groupby('seriesuid')
    args = [(seriesuid, group, luna_data, luna_segment, savepath, roi_size, resolution)
            for seriesuid, group in grouped]

    
    with Pool(processes=num_workers) as pool:
        pool.map(process_series, args)

if __name__ == '__main__':
    csv_path = 'D:\\HG\\3DCNN\DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection-master\\Luna_data\\candidates_V21.csv'
    luna_data = 'D:/HG/3DCNN/DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection-master/Luna_data/'
    luna_segment = 'D:/HG/3DCNN/LUNA/LUNA16/dataset/seg-lungs-LUNA16/'
    savepath = 'D:\\HG\\3DCNN\\classification\\class_npyP'
    preprocess_for_classification(csv_path, luna_data, luna_segment, savepath, roi_size=32, num_workers=10)



