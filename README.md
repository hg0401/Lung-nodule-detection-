# Lung-nodule-detection-
data LUNA16： https://luna16.grand-challenge.org/data/
 
Candidate region detection
Run prepare.py, output LUNA16 data can be found inside folder Preprocess_result_path, with saved images as _clean.npy, _label.npy , _spacing.npy, _extendbox.npy, _origin.npy for  training and data testing.
Run python train_detector_se.py to train the detector, then use NMS.py to eliminate redundant bounding boxes from the generated candidate boxes

False positive remove
Run preprocess.py ,Extract 32×32×32 voxel patches based on patient IDs and labels, then save them in _data.npz for subsequent training and testing.
Run KAN3-0427.py trains the network, while KAN3test.py eliminates false positive nodules
Run noduleCADEvaluationLUNA16.py and evaluate the results using the official LUNA16 FROC (Free-Response Receiver Operating Characteristic) method
