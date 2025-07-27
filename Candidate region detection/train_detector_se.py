import argparse
import torch.profiler
import os
import time
import numpy as np
# from setuptools.sandbox import save_path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from data_loader import LungNodule3Ddetector, collate
from importlib import import_module
import shutil
from utils import *
import sys

sys.path.append('../')
from split_combine import SplitComb
import pdb
from tqdm import tqdm

import torch
from torch.nn import DataParallel
from torch.backends import cudnn

from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from config_training import config as config_training
from torch.optim import lr_scheduler
import threading
from layers_se import acc, topkpbb

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='res18_se',
                    help='model')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='10', type=int, metavar='S',
                    help='save frequency')
# parser.add_argument('--resume', default='177.ckpt', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='004.ckpt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='U-Net0111_aRes_CMFA12_MC135123w/', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=1, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')






class EarlyStopping:  # 1010 新加早停法
    """Early stopping to stop the training when the loss does not improve after certain epochs."""

    def __init__(self, patience=40, verbose=False, delta=0, save_path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model)
            self.best_loss = val_loss

            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.save_path)


def main():
    print(f"Current memory allocated: {torch.cuda.memory_allocated()} bytes,main")  # 923

    global args
    args = parser.parse_args()
    best_tpr=0
    tpr_val=[]
    tpr_train=[]
    pre_train=[]
    loss_train=[]
    loss_val=[]
    pre_val=[]
    early_stopping = EarlyStopping(patience=40, verbose=True,save_path=os.path.join(args.save_dir, 'best_model.pth'))  # 修改patience

    torch.manual_seed(0)
    torch.cuda.set_device(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()
    start_epoch = args.start_epoch
    save_dir = args.save_dir

    # 使用ckpt
    if args.test==1:
        if args.resume:
           print("=> loading checkpoint '{}'".format(args.resume))
           checkpoint = torch.load(save_dir + 'detector_' + args.resume)
           start_epoch = checkpoint['epoch']
           net.load_state_dict(checkpoint['state_dict'])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log')
    if args.test != 1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f, os.path.join(save_dir, f))
    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)
    datadir = config_training['preprocess_result_path1']

    luna_train = np.load('./luna_train.npy')
    # luna_train = np.load('./luna_train02.npy')
    luna_test = np.load('./luna_test.npy')

    # luna_train = np.load('./LNDb_train_all.npy')
    # luna_test = np.load('./LNDb_val_all.npy')



    if args.test == 1:
        print("start test")
        margin = 32
        sidelen = 144
        split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])
        dataset = LungNodule3Ddetector(datadir, luna_test, config, phase='test', split_comber=split_comber)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate,
                                 pin_memory=False)
        test(test_loader, net, get_pbb, save_dir, config)
        return

    dataset = LungNodule3Ddetector(datadir, luna_train, config, phase='train')
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)



    dataset = LungNodule3Ddetector(datadir, luna_test, config, phase='val')
    val_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=args.workers, pin_memory=True)

    # optimizer = optim.Adadelta(net.parameters(), lr=args.lr, rho=0.95, eps=1e-8, weight_decay=args.weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # 修改学习率调度器为 ReduceLROnPlateau
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    print(f"Current memory allocated: {torch.cuda.memory_allocated()} bytes,beforetrain")  # 923

    for epoch in range(start_epoch, args.epochs + 1):
        # if epoch>5:
        #     optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
        # # 修改学习率调度器为 ReduceLROnPlateau
        #     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5)

        # 根据当前轮次选择优化器
        if epoch < 5:
            # 初期训练阶段，使用 Adam
            if isinstance(optimizer, optim.SGD):
                optimizer = optim.Adam(net.parameters(), lr=args.lr)
        elif 5 <= epoch <20:
            # 中期训练阶段，使用 SGD + Momentum
            if not isinstance(optimizer, optim.SGD):
                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
        else:
            # 后期训练阶段，使用 SGD
            if isinstance(optimizer, optim.SGD) and optimizer.param_groups[0]['momentum'] > 0:
                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0)
            elif isinstance(optimizer, optim.Adam):
                optimizer = optim.AdamW(net.parameters(), lr=args.lr)

        current_lr = scheduler.get_last_lr()[0]
        # print(f"Epoch {epoch}, Using optimizer: {optimizer.__class__.__name__}")
        print(f"Epoch {epoch}, current learning rate: {current_lr},Using optimizer: {optimizer.__class__.__name__}")
        trainloss,train_tpr,train_pre=train(train_loader, net, loss, epoch, optimizer, save_dir)

        # 验证模型
        valiloss ,val_tpr,val_pre= validate(val_loader, net, loss)
        print("**************valiloss {}".format(valiloss))

        # tpr_train.append(train_tpr)
        # tpr_val.append(val_tpr)
        #
        # loss_train.append(trainloss)
        # loss_val.append(valiloss)
        #
        # pre_train.append(train_pre)
        # pre_val.append(val_pre)
        tpr_train.append(round(train_tpr, 2))  # 保留两位小数
        tpr_val.append(round(val_tpr, 2))

        loss_train.append(round(trainloss, 5))  # 保留四位小数
        loss_val.append(round(valiloss, 5))

        pre_train.append(round(train_pre, 3))  # 保留三位小数
        pre_val.append(round(val_pre, 3))


        # 使用 ReduceLROnPlateau 调整学习率
        scheduler.step(valiloss)

        # 调用早停法，如果满足条件就退出
        early_stopping(valiloss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # 保存最优模型
        if early_stopping.best_loss >= valiloss:
            state_dict = net.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'epoch': epoch + 1,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'args': args
            }, os.path.join(save_dir, 'detector_%03d.ckpt' % epoch))
            print("save model on epoch %d" % epoch)
            print("Model saved to: {}".format(save_dir))

        if val_tpr >= best_tpr:
            best_tpr=val_tpr
            state_dict = net.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'epoch': epoch + 1,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'args': args
            }, os.path.join(save_dir, 'detector_%03d.ckpt' % epoch))
            print("save model on epoch %d" % epoch)
            print("Model saved to: {}".format(save_dir))
    # 绘制 TPR 曲线
    print(tpr_train)
    print(tpr_val)
    print(loss_train)
    print(loss_val)
    print(pre_train)
    print(pre_val)
    #
    # plt.figure(figsize=(10, 5))  # 设置图像大小
    # plt.plot(range(1, args.epochs+2), tpr_train, label='TPR-train')
    # plt.plot(range(1, args.epochs + 2), tpr_val, label='TPR-val')
    # plt.xlabel('Epoch')
    # plt.ylabel('True Positive Rate (TPR)')
    # plt.title('TPR vs Epoch')
    # plt.legend()
    # plt.grid(True)
    # # plt.savefig('tpr_vs_epoch.png')
    #
    # plt.figure(figsize=(10, 5))  # 设置图像大小
    # plt.plot(range(1, args.epochs + 2), loss_train, label='Loss-train')
    # plt.plot(range(1, args.epochs + 2), loss_val, label='Loss-val')
    # plt.xlabel('Epoch')
    # plt.ylabel('LOSS')
    # plt.title('LOSS vs Epoch')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

def train(data_loader, net, loss, epoch, optimizer,  save_dir):
    start_time = time.time()
    # print(f"Start of training epoch {epoch}: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")  # 开始时的显存923

    net.train()
    # lr = get_lr(epoch)
    # for param_group in optimizer.param_groups:
    #     print(f"Current learning rate: {param_group['lr']}")#927打印学习率
    #     param_group['lr'] = lr

    metrics = []
    with tqdm(total=len(data_loader), desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as pbar:
        for i, (data, target, coord) in enumerate(data_loader):
            data, target, coord = data.cuda(), target.cuda(), coord.cuda()
            # print(coord.size())
            optimizer.zero_grad()
            # print(coord.size())
            output = net(data, coord)
            loss_output = loss(output, target)
            loss_output[0].backward()
            optimizer.step()

            loss_output_cpu = [item.item() if isinstance(item, torch.Tensor) else item for item in loss_output]
            metrics.append(loss_output_cpu)

            pbar.set_postfix(loss=loss_output_cpu[0])  # 更新进度条的显示，显示当前损失
            pbar.update(1)  # 更新进度条

    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    #  print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print('precision',100.0*np.sum(metrics[:, 6]) /(np.sum(metrics[:, 9])-np.sum(metrics[:, 8])+np.sum(metrics[:, 6])))
    print("****************************")
    print(f"Epoch {epoch}, avg loss {np.mean(metrics[:, 0])}")
    return np.mean(metrics[:, 0]),100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),100.0*np.sum(metrics[:, 6]) /(np.sum(metrics[:, 9])-np.sum(metrics[:, 8])+np.sum(metrics[:, 6]))


def validate(data_loader, net, loss):  # 验证函数
    start_time = time.time()
    # print(f"Start of validation: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")  # 开始时的显存923
    net.eval()

    metrics = []
    with torch.no_grad():  # 后加禁用梯度计算923
        for i, (data, target, coord) in enumerate(data_loader):
            # data = Variable(data.cuda(non_blocking = True))
            # target = Variable(target.cuda(non_blocking = True))
            # coord = Variable(coord.cuda(non_blocking = True))
            data = data.cuda(non_blocking=True)  # 后加3 修改了上面直接使用 .cuda()923
            target = target.cuda(non_blocking=True)
            coord = coord.cuda(non_blocking=True)

            output = net(data, coord)
            loss_output = loss(output, target, train=False)

            # loss_output[0] = loss_output[0].item()
            # metrics.append(loss_output)

            loss_output_cpu = [item.item() if isinstance(item, torch.Tensor) else item for item in loss_output]  # 3.后加2

            metrics.append(loss_output_cpu)
            torch.cuda.empty_cache()  # 清空CUDA缓存923
    # print(f"End of validation: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")  # 结束时的显存923
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    # 真正率   真负率    正样本总数     负样本总数
    print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    # 总损失       分类损失       回归损失，三个
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print('precision',100.0 * np.sum(metrics[:, 6]) / (np.sum(metrics[:, 9]) - np.sum(metrics[:, 8]) + np.sum(metrics[:, 6])))
    torch.cuda.empty_cache()  # 清空CUDA缓存923
    return np.mean(metrics[:, 0]),100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),100.0*np.sum(metrics[:, 6]) /(np.sum(metrics[:, 9])-np.sum(metrics[:, 8])+np.sum(metrics[:, 6]))


def test(data_loader, net, get_pbb, save_dir, config):
    start_time = time.time()
    save_dir = os.path.join(save_dir, 'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber

    print(f"Start loop with {len(data_loader)} items.")  # 后加
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        # 打印target的内容和形状后加1
        print(f"Target content: {target}")

        target = [np.asarray(t, np.float32) for t in target]
        print(f"Target[0] content: {target[0]}")
        lbb = target[0]
        print(f"lbb content: {lbb}")
        nzhw = nzhw[0]
        # name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1].split('_clean')[0]
        name = data_loader.dataset.filenames[i_name].split('/')[-1].split('_')[0]
        data = data[0][0]
        coord = coord[0][0]

        print(f"Processing {name}, loop index {i_name}")  # 后加 打印当前处理的图像名称和循环索引

        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = args.n_test
        print(data.size())

        splitlist = list(range(0, len(data) + 1, n_per_run))
        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        with torch.no_grad():  # 后加禁用梯度计算  925
            for i in range(len(splitlist) - 1):

                input = Variable(data[splitlist[i]:splitlist[i + 1]].cuda(non_blocking=True))

                inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]].cuda(non_blocking=True))
                if isfeat:
                    output, feature = net(input, inputcoord)
                    featurelist.append(feature.data.cpu().numpy())
                else:
                    output = net(input, inputcoord)

                outputlist.append(output.data.cpu().numpy())

                del input, inputcoord, output
                # torch.cuda.empty_cache()

        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)
        if isfeat:
            feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]
            # feature = split_comber.combine(feature,sidelen)[...,0]
            feature = split_comber.combine(feature, nzhw)[..., 0]

        thresh = -3
        # thresh=0.5
        pbb, mask = get_pbb(output, thresh, ismask=True)
        if isfeat:
            feature_selected = feature[mask[0], mask[1], mask[2]]
            # 如果配置中指定输出特征，还会保存每个图像的特征选择结果（feature_selected），保存为.npy文件
            np.save(os.path.join(save_dir, name + '_feature.npy'), feature_selected)
        # tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        # print([len(tp),len(fp),len(fn)])
        # print("############")
        # print(tp,fp,fn)
        # namelist.append(name)  # 后加2 确保namelist被更新
        # print(f"Updated namelist: {namelist}")  # 打印更新后的namelist
        print([i_name, name])

        # 自写def topkpbb(pbb,lbb,nms_th,detect_th,topk=30):后加4
        # tp,fp,fn=topkpbb(pbb,lbb,0.1,0.1,30)
        # precision, recall, f1_score =calculate_accuracy(tp, fp, fn)
        # print(precision, recall, f1_score )

        print(f"pbb content: {pbb}")  # 后加2
        print(f"pbb shape: {pbb.shape}")
        # 预测边界框保存
        np.save(os.path.join(save_dir, name + '_pbb.npy'), pbb)
        # 真实边界框保存
        # np.save(os.path.join(save_dir, name+'_lbb.npy'), lbb)

        print(f"lbb content: {lbb}")  # 后加2
        print(f"lbb shape: {lbb.shape}")
        if lbb.shape[0] > 0:  # 后加5
            np.save(os.path.join(save_dir, name + '_lbb.npy'), lbb)
            print(f"Saved {name}'s true bounding boxes.")
        else:
            print(f"No true bounding boxes to save for {name}.")

        print("########################")
    #     所有处理过的图像名称
    # print(f"End loop, namelist length: {len(namelist)}")#后加2
    # print(f"Saving namelist to {os.path.join(save_dir, 'namelist.npy')}")
    # np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    if os.path.exists(os.path.join(save_dir, 'namelist.npy')):
        # 文件存在，加载现有数据
        existing_data = np.load(os.path.join(save_dir, 'namelist.npy'))
        # 保存数据
        # 将整数转换为数组，以便追加
        new_data_array = np.array([name])
        new_data = np.append(existing_data, new_data_array)
        np.save(os.path.join(save_dir, 'namelist.npy'), new_data)
    else:
        # 文件不存在，直接保存新数据
        np.save(os.path.join(save_dir, 'namelist.npy'), name)
    end_time = time.time()

    print('elapsed time is %3.2f seconds' % (end_time - start_time))


def singletest(data, net, config, splitfun, combinefun, n_per_run, margin=64, isfeat=False):
    z, h, w = data.size(2), data.size(3), data.size(4)
    print(data.size())
    data = splitfun(data, config['max_stride'], margin)
    data = Variable(data.cuda(non_blocking=True), requires_grad=False)
    splitlist = range(0, args.split + 1, n_per_run)
    outputlist = []
    featurelist = []
    for i in range(len(splitlist) - 1):
        if isfeat:
            output, feature = net(data[splitlist[i]:splitlist[i + 1]])
            featurelist.append(feature)
        else:
            output = net(data[splitlist[i]:splitlist[i + 1]])
        output = output.data.cpu().numpy()
        outputlist.append(output)

    output = np.concatenate(outputlist, 0)
    output = combinefun(output, z / config['stride'], h / config['stride'], w / config['stride'])
    if isfeat:
        feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])
        feature = combinefun(feature, z / config['stride'], h / config['stride'], w / config['stride'])
        return output, feature
    else:
        return output


if __name__ == '__main__':
    main()
