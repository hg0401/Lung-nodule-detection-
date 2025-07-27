import numpy as np
from noduleCADEvaluationLUNA16 import noduleCADEvaluation
import os 
import csv 
from multiprocessing import Pool
import functools
import SimpleITK as sitk
# from config_testing import config
from config_training import config#后加
from layers import nms

annotations_filename = './labels/annos.csv'
annotations_excluded_filename = './labels/excluded.csv'# path for excluded annotations for the fold
seriesuids_filename = './labels/luna_ID.csv'# path for seriesuid for the fold
datapath = config['luna_data']
sideinfopath = '../DeepSEED-3D-ConvNets-for-Pulmonary-Nodule-Detection-master/Preprocess_result_path1219_A/'
# NMS 是一种后处理方法，用于去除检测中重复的或重叠的候选框。nmsthresh 控制了候选框重叠的程度，通常介于 0 到 1 之间。较低的值意味着更多的重叠框会被抑制。
nmsthresh = 0.1

bboxpath = './U-Net1219_B//bbox/' #for baseline  要修改
frocpath = './U-Net1219_B/baseline_se_focal_newparam/bbox/nms' + str(nmsthresh) + '/' #_focal  要修改

outputdir = './bboxoutput/se_focal/nms' + str(nmsthresh) + '/'

#detp = [0.3, 0.4, 0.5, 0.6, 0.7]
detp = [0.3]

# nprocess = 38#4
nprocess = 16# 后加927
# firstline = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']
firstline = ['seriesuid', 'coordX', 'coordY', 'coordZ','diameter_mm', 'probability']#"后加

def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def convertcsv(bboxfname, bboxpath, detp):
    print("3333333333333333")
    resolution = np.array([1, 1, 1])
    
    origin = np.load(sideinfopath+bboxfname[:-8]+'_origin.npy', mmap_mode='r')
    spacing = np.load(sideinfopath+bboxfname[:-8]+'_spacing.npy', mmap_mode='r')
    extendbox = np.load(sideinfopath+bboxfname[:-8]+'_extendbox.npy', mmap_mode='r')
    
    pbb = np.load(bboxpath+bboxfname, mmap_mode='r')
    diam = pbb[:,-1]
    
    check = sigmoid(pbb[:,0]) > detp
    pbbold = np.array(pbb[check])
#    pbbold = np.array(pbb[pbb[:,0] > detp])
    pbbold = np.array(pbbold[pbbold[:,-1] > 3])  # add new 9 15
    
    pbb = nms(pbbold, nmsthresh)
    pbb = np.array(pbb[:, :-1])
    pbb[:, 1:] = np.array(pbb[:, 1:] + np.expand_dims(extendbox[:,0], 1).T)
    pbb[:, 1:] = np.array(pbb[:, 1:] * np.expand_dims(resolution, 1).T / np.expand_dims(spacing, 1).T)
    
    pos = VoxelToWorldCoord(pbb[:, 1:], origin, spacing)
    rowlist = []
    
    for nk in range(pos.shape[0]): # pos[nk, 2], pos[nk, 1], pos[nk, 0]
        rowlist.append([bboxfname[:-8], pos[nk, 2], pos[nk, 1], pos[nk, 0], diam[nk], 1/(1+np.exp(-pbb[nk,0]))])

    print("4444444444444444444444")
    return rowlist

def getfrocvalue(results_filename, outputdir):
    return noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,outputdir)

def getcsv(detp):
    if not os.path.exists(frocpath):
        os.makedirs(frocpath)
        
    for detpthresh in detp:
        print ('detp', detpthresh)
        f = open(frocpath + 'predanno'+ str(detpthresh) + '.csv', 'w',newline='')
        fwriter = csv.writer(f)
        fwriter.writerow(firstline)
        fnamelist = []
        for fname in os.listdir(bboxpath):
            if fname.endswith('_pbb.npy'):
                fnamelist.append(fname)
                # print fname
                # for row in convertcsv(fname, bboxpath, k):
                    # fwriter.writerow(row)
        # # return
        print(len(fnamelist))
        print("11111111111111")
        predannolist = p.map(functools.partial(convertcsv, bboxpath=bboxpath, detp=detpthresh), fnamelist)
        print("2222222222222")
        # print len(predannolist), len(predannolist[0])
        for predanno in predannolist:
            # print predanno
            for row in predanno:
                # print row
                fwriter.writerow(row)
        f.close()

def getfroc(detp):
    
    predannofnamalist = []
    outputdirlist = []
    for detpthresh in detp:
        predannofnamalist.append(outputdir + 'predanno'+ str(detpthresh) + '.csv')
        outputpath = outputdir + 'predanno'+ str(detpthresh) +'/'
        outputdirlist.append(outputpath)
        
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
#    froclist = p.map(getfrocvalue, predannofnamalist, outputdirlist)
        
    froclist = []
    for i in range(len(predannofnamalist)):
        froclist.append(getfrocvalue(predannofnamalist[i], outputdirlist[i]))
    
    np.save(outputdir+'froclist.npy', froclist)
           
        
if __name__ == '__main__':
    p = Pool(nprocess)
    getcsv(detp)
    
#    getfroc(detp)
    p.close()
    print('finished!')
    
        
        
        
        
        
        