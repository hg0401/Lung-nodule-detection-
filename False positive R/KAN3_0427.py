import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
from torch.nn.parallel import DataParallel
import argparse
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import torchvision.transforms.functional as TF
from scipy.ndimage import rotate

EPOCHS = 200

# EarlyStopping 
class EarlyStopping:
    def __init__(self, patience=40, verbose=False, delta=0, save_path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_f1 = None
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path

    def __call__(self, f1, model):
        score = -f1
        if self.best_f1 is None:
            self.best_f1 = f1
            self.save_checkpoint(f1, model)
        elif score > -self.best_f1 + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(f1, model)
            self.best_f1 = f1
            self.counter = 0

    def save_checkpoint(self, f1, model):
        if self.verbose:
            print(f"Validation F1 increased ({self.best_f1:.6f} --> {f1:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.save_path)


class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, spline_order=2, grid_size=3, use_multiplication=False):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.use_multiplication = use_multiplication

        # Parameter Initialization
        self.w_b = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.w_b, mode='fan_in', nonlinearity='relu')  # An initialization strategy with improved stability
        self.w_s = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.w_s, mode='fan_in', nonlinearity='relu')
        self.coefficients = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.01)  # Reduce the initial values
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
            # Prevent division by zero
            denom_left = torch.where(denom_left.abs() < 1e-6, torch.ones_like(denom_left) * 1e-6, denom_left)
            denom_right = torch.where(denom_right.abs() < 1e-6, torch.ones_like(denom_right) * 1e-6, denom_right)
            left = (x - grid[..., :-k - 1]) / denom_left
            right = (grid[..., k + 1:] - x) / denom_right
            basis = left * basis[..., :-1] + right * basis[..., 1:]
        return basis[..., :self.grid_size].clamp(-10, 10)  # Restrict the output range

    def forward(self, x):
       #"Forward propagation"  
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

        #Spline Output
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

        # Add bias
        if is_3d:
            output = output + self.bias.view(1, 1, -1)
        else:
            output = output + self.bias.view(1, -1)

        if is_3d:
            output = output.transpose(1, 2).view(batch_size, self.out_features, D, H, W)

        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            print("Warning: NaN or Inf in final output")

        return output.clamp(-10, 10)  # Restrict the final output range


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



class KANet3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Encoder section
        self.conv1 = ConvKAN3D(1, 16, pool=False)
        self.conv2 = ConvKAN3D(16, 32)
        self.conv3 = ConvKAN3D(32, 64)
        self.conv4 = ConvKAN3D(64, 128)  
        self.conv5 = ConvKAN3D(128, 256)  

        #Attention Mechanism
        self.attn1 = SelfKAGNtention3D(64)
        self.attn2 = SelfKAGNtention3D(256)  # Deep Attention

        # Decoder section
        self.up5 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.up4 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.up3 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.up2 = nn.ConvTranspose3d(32, 16, 2, stride=2)

        #Feature Fusion Enhancement
        self.fuse_conv = nn.Conv3d(16 + 16 + 32 + 64 + 256, 256, 1)  # Corrected channels: 384 -> 256

        # Classification Head Adjustment
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.5)  # Enhanced Regularization
        self.fc1 = KANLayer(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c3 = self.attn1(c3)  # Middle-level attention
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c5 = self.attn2(c5)  #Deep Attention

        # Decoding process
        up5 = self.up5(c5)
        up4 = self.up4(up5 + c4)  # skip connection
        up3 = self.up3(up4 + c3)  # skip connection
        up2 = self.up2(up3 + c2)  # skip connection

        # Multi-scale feature fusion
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

# class of datasets
class LungNoduleDataset(Dataset):
    def __init__(self, data_dir, series_list, training=True):
        self.data_dir = data_dir
        self.series_list = series_list
        self.training = training
        self.rois = []
        self.labels = []

        print("Loading data into memory...")
        for seriesuid in series_list:
            npz_path = os.path.join(data_dir, f'{seriesuid}_data.npz')
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                rois = data['rois']
                #Ensure that the ROIs have a channel dimension.
                if rois.ndim == 4:  # [N, 32, 32, 32]
                    rois = rois[:, np.newaxis, :, :, :]  # [N, 1, 32, 32, 32]
                self.rois.append(rois)
                self.labels.append(data['labels'])
        self.rois = np.concatenate(self.rois, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        neg_indices = np.where(self.labels == 0)[0]
        pos_indices = np.where(self.labels == 1)[0]
        np.random.shuffle(neg_indices)
        selected_neg_indices = neg_indices[:len(pos_indices) * 20]
        final_indices = np.concatenate([selected_neg_indices, pos_indices])
        self.rois = self.rois[final_indices]
        self.labels = self.labels[final_indices]

        # Only perform repeated oversampling on the training set.
        if self.training:
            print("Applying 2x oversampling to positive class (training set only)...")
            pos_indices = np.where(self.labels == 1)[0]
            pos_rois = self.rois[pos_indices]
            pos_labels = self.labels[pos_indices]
            self.rois = np.concatenate([self.rois, pos_rois], axis=0)
            self.labels = np.concatenate([self.labels, pos_labels], axis=0)
            print(f"After oversampling (training): {len(self.labels)} samples, label distribution: {np.bincount(self.labels)}")
        else:
            print(f"No oversampling applied (validation): {len(self.labels)} samples, label distribution: {np.bincount(self.labels)}")

        print(f"self.rois shape: {self.rois.shape}")  #Debug output

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        roi = torch.tensor(self.rois[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        # Ensure the ROI has a channel dimension [1, 32, 32, 32].
        if roi.ndim == 3:  # [32, 32, 32]
            roi = roi.unsqueeze(0)  # [1, 32, 32, 32]
        roi = (roi - 128) / 128  # Normalize to [-1, 1]
        roi = roi.clamp(-1, 1)
        return roi, label

    def get_labels(self):
        return self.labels

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, weights=None):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if weights is not None:
            focal_loss = focal_loss * weights
        return focal_loss.mean()

# WarmupScheduler
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, max_lr, total_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.max_lr * self.current_epoch / self.warmup_epochs
        else:
            lr = self.max_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Training and validation functions
def train(data_loader, net, criterion, epoch, optimizer, save_dir):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    scaler = GradScaler()
    dropout_rate = max(0.5 - epoch * 0.01, 0.1)
    net.module.dropout.p = dropout_rate

    with tqdm(total=len(data_loader), desc=f"Epoch {epoch}/{EPOCHS}", unit="batch") as pbar:
        for i, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix(loss=running_loss / (i + 1), acc=100. * correct / total)
            pbar.update(1)

    epoch_loss = running_loss / len(data_loader)
    epoch_acc = 100. * correct / total
    print(f"Train Epoch {epoch}: Loss {epoch_loss:.4f}, Accuracy {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc

def validate(data_loader, net, criterion):
    net.eval()
    running_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            probs = F.softmax(outputs, dim=1)[:, 1]
            total += targets.size(0)
            all_preds.extend((probs > 0.5).long().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    thresholds = np.linspace(0.4, 0.7, 50)
    best_f1, best_thresh = 0, 0.5
    for thresh in thresholds:
        preds = (np.array(all_probs) > thresh).astype(int)
        f1 = f1_score(all_targets, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    all_preds = (np.array(all_probs) > best_thresh).astype(int)
    correct = (all_preds == all_targets).sum()

    val_loss = running_loss / len(data_loader)
    val_acc = 100. * correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    auc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.0
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"Validation: Loss {val_loss:.4f}, Accuracy {val_acc:.2f}%, Precision {precision:.4f}, Recall {recall:.4f}, F1 {f1:.4f}, Specificity {specificity:.4f}, AUC {auc:.4f}, Best Threshold {best_thresh:.4f}")
    print("Confusion Matrix:\n", cm)
    return val_loss, val_acc, f1

# main
def main0():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='D:\\HG\\3DCNN\\classification\\models', type=str)
    parser.add_argument('--data_dir', default='D:\\HG\\3DCNN\\classification\\class_npyP', type=str)
    parser.add_argument('--train_series_file', default='D:\\HG\\3DCNN\\LUNA_HDF5\\luna_train.npy', type=str)
    parser.add_argument('--val_series_file', default='D:\\HG\\3DCNN\\LUNA_HDF5\\luna_test.npy', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=EPOCHS, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()
    torch.manual_seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.set_device(0)

    net = get_kanet3d(num_classes=2).cuda()
    net = DataParallel(net)

    train_series = np.load(args.train_series_file).tolist()
    val_series = np.load(args.val_series_file).tolist()
    print(f"Loaded {len(train_series)} seriesuids for training.")
    print(f"Loaded {len(val_series)} seriesuids for validation.")

    train_dataset = LungNoduleDataset(args.data_dir, train_series, training=True)
    val_dataset = LungNoduleDataset(args.data_dir, val_series, training=False)

    train_labels = train_dataset.get_labels()
    neg_count, pos_count = np.bincount(train_labels)
    class_weights = [1.0, neg_count / pos_count]
    sampler = WeightedRandomSampler([class_weights[label] for label in train_labels], len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=20, max_lr=args.lr, total_epochs=args.epochs)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    criterion = FocalLoss(alpha=pos_count / (neg_count + pos_count), gamma=3).cuda()
    early_stopping = EarlyStopping(patience=20, verbose=True, save_path=os.path.join(args.save_dir, 'best_model.pth'))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_losses, val_losses, val_accs, val_f1s = [], [], [], []

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_acc = train(train_loader, net, criterion, epoch, optimizer, args.save_dir)
        val_loss, val_acc, val_f1 = validate(val_loader, net, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        if epoch < 20:
            warmup_scheduler.step()
        else:
            scheduler.step(val_f1)

        early_stopping(val_f1, net)
        if early_stopping.early_stop:
            print("Early stopping triggered after first stage")
            break

        if val_f1 >= early_stopping.best_f1:
            state_dict = net.module.state_dict()
            torch.save({'epoch': epoch + 1, 'state_dict': state_dict, 'args': args},
                       os.path.join(args.save_dir, f'classifier_epoch_{epoch:03d}.ckpt'))
            print(f"Model saved at epoch {epoch} with val_f1 {val_f1:.4f}")

    print("Starting second stage: fine-tuning fc layer...")
    best_checkpoint = torch.load(early_stopping.save_path)
    state_dict = {k.replace('module.', ''): v for k, v in best_checkpoint.items()}
    net.module.load_state_dict(state_dict)

    for param in net.module.parameters():
        param.requires_grad = False
    for param in net.module.fc1.parameters():
        param.requires_grad = True
    for param in net.module.fc2.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(list(net.module.fc1.parameters()) + list(net.module.fc2.parameters()),
                           lr=0.0001, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=15, verbose=True, save_path=os.path.join(args.save_dir, 'best_model_finetuned.pth'))

    for epoch in range(100):
        train_loss, train_acc = train(train_loader, net, criterion, epoch, optimizer, args.save_dir)
        val_loss, val_acc, val_f1 = validate(val_loader, net, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        scheduler.step()
        early_stopping(val_f1, net)
        if early_stopping.early_stop:
            print("Early stopping triggered after second stage")
            break

        if val_f1 >= early_stopping.best_f1:
            state_dict = net.module.state_dict()
            torch.save({'epoch': epoch + 1, 'state_dict': state_dict, 'args': args},
                       os.path.join(args.save_dir, f'finetuned_epoch_{epoch:03d}.ckpt'))
            print(f"Finetuned model saved at epoch {epoch} with val_f1 {val_f1:.4f}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(val_f1s, label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='D:\\HG\\3DCNN\\classification\\models', type=str)
    parser.add_argument('--data_dir', default='D:\\HG\\3DCNN\\classification\\class_npyP', type=str)
    parser.add_argument('--train_series_file', default='D:\\HG\\3DCNN\\LUNA_HDF5\\luna_train.npy', type=str)
    parser.add_argument('--val_series_file', default='D:\\HG\\3DCNN\\LUNA_HDF5\\luna_test.npy', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=EPOCHS, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--resume', default=None, type=str, help='Path to checkpoint to resume from (e.g., best_model.pth)')
    # python KAN2.py --resume D:\HG\3DCNN\classification\models\best_model.pth
    args = parser.parse_args()
    torch.manual_seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.set_device(0)

    net = get_kanet3d(num_classes=2).cuda()
    net = DataParallel(net)

    train_series = np.load(args.train_series_file).tolist()
    val_series = np.load(args.val_series_file).tolist()
    print(f"Loaded {len(train_series)} seriesuids for training.")
    print(f"Loaded {len(val_series)} seriesuids for validation.")

    train_dataset = LungNoduleDataset(args.data_dir, train_series, training=True)
    val_dataset = LungNoduleDataset(args.data_dir, val_series, training=False)

    train_labels = train_dataset.get_labels()
    neg_count, pos_count = np.bincount(train_labels)
    class_weights = [1.0, neg_count / pos_count]
    sampler = WeightedRandomSampler([class_weights[label] for label in train_labels], len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=20, max_lr=args.lr, total_epochs=args.epochs)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    criterion = FocalLoss(alpha=pos_count / (neg_count + pos_count), gamma=3).cuda()
    early_stopping = EarlyStopping(patience=20, verbose=True, save_path=os.path.join(args.save_dir, 'best_model.pth'))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_losses, val_losses, val_accs, val_f1s = [], [], [], []

    # If resume is specified, load the checkpoint and skip the first stage.
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        best_checkpoint = torch.load(args.resume)
        state_dict = best_checkpoint if 'state_dict' not in best_checkpoint else best_checkpoint['state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        net.module.load_state_dict(state_dict)
    else:
        # First-stage training
        for epoch in range(args.start_epoch, args.epochs):
            train_loss, train_acc = train(train_loader, net, criterion, epoch, optimizer, args.save_dir)
            val_loss, val_acc, val_f1 = validate(val_loader, net, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_f1s.append(val_f1)

            if epoch < 20:
                warmup_scheduler.step()
            else:
                scheduler.step(val_f1)

            early_stopping(val_f1, net)
            if early_stopping.early_stop:
                print("Early stopping triggered after first stage")
                break

            if val_f1 >= early_stopping.best_f1:
                state_dict = net.module.state_dict()
                torch.save({'epoch': epoch + 1, 'state_dict': state_dict, 'args': args},
                           os.path.join(args.save_dir, f'classifier_epoch_{epoch:03d}.ckpt'))
                print(f"Model saved at epoch {epoch} with val_f1 {val_f1:.4f}")

    # Phase 2: Fine-tuning the FC layer
    print("Starting second stage: fine-tuning fc layer...")
    if not args.resume:  # If it is not a resume, load from the best model of the first stage.
        best_checkpoint = torch.load(early_stopping.save_path)
        state_dict = best_checkpoint if 'state_dict' not in best_checkpoint else best_checkpoint['state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        net.module.load_state_dict(state_dict)

    for param in net.module.parameters():
        param.requires_grad = False
    for param in net.module.fc1.parameters():
        param.requires_grad = True
    for param in net.module.fc2.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(list(net.module.fc1.parameters()) + list(net.module.fc2.parameters()),
                           lr=0.0001, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=15, verbose=True, save_path=os.path.join(args.save_dir, 'best_model_finetuned.pth'))

    for epoch in range(100):
        train_loss, train_acc = train(train_loader, net, criterion, epoch, optimizer, args.save_dir)
        val_loss, val_acc, val_f1 = validate(val_loader, net, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        scheduler.step()
        early_stopping(val_f1, net)
        if early_stopping.early_stop:
            print("Early stopping triggered after second stage")
            break

        if val_f1 >= early_stopping.best_f1:
            state_dict = net.module.state_dict()
            torch.save({'epoch': epoch + 1, 'state_dict': state_dict, 'args': args},
                       os.path.join(args.save_dir, f'finetuned_epoch_{epoch:03d}.ckpt'))
            print(f"Finetuned model saved at epoch {epoch} with val_f1 {val_f1:.4f}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(val_f1s, label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()

