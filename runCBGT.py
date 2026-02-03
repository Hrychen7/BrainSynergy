import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
# 确保从 model.CBGT 导入 CBGTNet
from model.CBGT import CBGTNet as net
from torch.utils import data
import torch.nn.functional as F
import sys

# ================= 0. Triplet Loss 定义 (保持您的原有逻辑) =================
class MyTriplet_loss(nn.Module):
    def __init__(self, margin=1.0):
        super(MyTriplet_loss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-16).sqrt()
        
        mask = self.get_mask(targets)
        ap_dist = dist.unsqueeze(2)
        an_dist = dist.unsqueeze(1)
        triplet_loss = ap_dist - an_dist + self.margin
        
        triplet_loss = torch.multiply(mask, triplet_loss)
        triplet_loss = torch.maximum(triplet_loss, torch.tensor(0.0).to(inputs.device))
        num_triplets = torch.sum((triplet_loss > 1e-16).float())
        return torch.sum(triplet_loss) / (num_triplets + 1e-16)

    def get_mask(self, targets):
        targets_equal = targets.unsqueeze(0).eq(targets.unsqueeze(1))
        i_equal_j = targets_equal.unsqueeze(2)
        i_equal_k = targets_equal.unsqueeze(1)
        mask = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))
        return mask.float()

# ================= 1. 数据集类 =================
class ADNIDataset(data.Dataset):
    def __init__(self, npz_path, task='MCI_AD'):
        alldata = np.load(npz_path)
        raw_data = alldata['data']
        raw_labels = alldata['label'].squeeze()

        if task == 'NC_MCI':
            target_labels = (0, 1)
        elif task == 'MCI_AD':
            target_labels = (1, 2)
        elif task == 'NC_AD':
            target_labels = (0, 2)
        elif task == 'ASD_NC': # 适配 ABIDE 任务
            target_labels = (0, 1)
        else:
            raise ValueError(f"Task {task} not supported")

        mask = (raw_labels == target_labels[0]) | (raw_labels == target_labels[1])
        self.data = raw_data[mask]
        
        new_labels = raw_labels[mask]
        final_labels = np.zeros_like(new_labels)
        final_labels[new_labels == target_labels[1]] = 1
        self.labels = final_labels.astype(np.int64)
        
        # 记录样本分割点用于保持原始的数据分块逻辑 (Block 1 / Block 2)
        diff = np.diff(raw_labels)
        split_points = np.where(diff < -0.5)[0]
        raw_split_idx = split_points[0] + 1 if len(split_points) > 0 else len(raw_labels) // 2
        self.split_idx = np.sum(mask[:raw_split_idx])
        
        print(f"[{task}] 初始化完成 | 样本总数: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ================= 2. 训练与评估逻辑 =================
def train_epoch(model, criterion_ce, criterion_tri, optimizer, loader, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, targets in loader:
        data, targets = data.to(device).float(), targets.to(device)
        # CBGT 输入期望: [B, ROI, ROI*2]
        data = data.view(data.size(0), data.size(1), -1) 
        
        embedding, probs = model(data)
        
        # 论文主要使用交叉熵，这里保留您的 Triplet Loss 作为正则项
        loss = criterion_ce(probs, targets) + 0.5 * criterion_tri(embedding, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (probs.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device).float(), targets.to(device)
            data = data.view(data.size(0), data.size(1), -1)
            
            _, probs = model(data)
            
            # probs 已经是 Softmax 后的结果 (在 CBGTNet 内部处理)
            all_preds.extend(probs.argmax(1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    acc = np.mean(all_preds == all_targets) * 100
    
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    try: auc = roc_auc_score(all_targets, all_probs)
    except: auc = 0
    
    return acc, sen, spe, auc

# ================= 3. 主程序 =================
def run_experiment(data_path, task, model_name, base_root, batch_size=32, lr=1e-4, epochs=200, device='cuda', seed=42):
    final_output_dir = os.path.join(base_root, task, model_name)
    os.makedirs(final_output_dir, exist_ok=True)
    
    dataset = ADNIDataset(data_path, task=task)
    indices = np.arange(len(dataset))
    
    # 按照您的原始逻辑进行 Block 划分
    part1_idx = indices[:dataset.split_idx]
    part2_idx = indices[dataset.split_idx:]
    
    # 5折交叉验证
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    p1_folds = list(skf.split(part1_idx, dataset.labels[part1_idx]))
    p2_folds = list(skf.split(part2_idx, dataset.labels[part2_idx]))

    fold_results = [] 

    for fold in range(10):
        print(f"\n--- {model_name} | {task} | Fold {fold} ---")
        train_idx = np.concatenate([part1_idx[p1_folds[fold][0]], part2_idx[p2_folds[fold][0]]])
        test_idx = np.concatenate([part1_idx[p1_folds[fold][1]], part2_idx[p2_folds[fold][1]]])
        
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size)

        # 初始化 CBGT 模型
        model = net(
            roi_num=100,          # 确保与您的 npz 数据 ROI 维度一致
            num_classes=2, 
            depth=2,              # 论文中的 L 层
            heads=4, 
            dropout=0.5, 
            pooling_ratio=0.7     # Pooling 保留比例
        ).to(device)

        # 论文建议学习率为 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        c1 = nn.CrossEntropyLoss()
        c2 = MyTriplet_loss(margin=0.8)

        fold_dir = os.path.join(final_output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        log_file = os.path.join(fold_dir, "train_log.txt")
        
        best_acc = 0
        best_metrics = None

        for epoch in range(epochs):
            t_loss, t_acc = train_epoch(model, c1, c2, optimizer, train_loader, device)
            v_acc, v_sen, v_spe, v_auc = evaluate(model, test_loader, device)
            
            if v_acc > best_acc:
                best_acc = v_acc
                best_metrics = [v_acc, v_sen, v_spe, v_auc]
                torch.save(model.state_dict(), os.path.join(fold_dir, f"{model_name}_best.pth"))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:03d} | Loss: {t_loss:.4f} | Acc: {v_acc:.2f}% | AUC: {v_auc:.4f}")

            with open(log_file, 'a') as f:
                f.write(f"Ep {epoch+1:03d} | Loss: {t_loss:.4f} | TestAcc: {v_acc:.2f}% | AUC: {v_auc:.4f}\n")

        fold_results.append(best_metrics)
        print(f"Fold {fold} Best Acc: {best_acc:.2f}%")

    # ================= 4. 计算均值与标准差 =================
    fold_results = np.array(fold_results)
    means = np.mean(fold_results, axis=0)
    stds = np.std(fold_results, axis=0)
    
    summary_path = os.path.join(final_output_dir, "final_summary.txt")
    metrics_names = ["Accuracy", "Sensitivity", "Specificity", "AUC"]
    
    with open(summary_path, 'w') as f:
        f.write(f"CBGT Experiment Summary on {task}\n")
        f.write("="*50 + "\n")
        f.write(f"{'Metric':<15} | {'Mean':<10} | {'Std':<10}\n")
        f.write("-" * 50 + "\n")
        for name, m, s in zip(metrics_names, means, stds):
            unit = "%" if name == "Accuracy" else ""
            f.write(f"{name:<15} | {m:>8.2f}{unit} | {s:>8.2f}{unit}\n")
        f.write("="*50 + "\n")
    
    print(f"\n实验完成！结果已汇总至: {summary_path}")

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    current_task = sys.argv[1] if len(sys.argv) > 1 else 'MCI_AD'
    
    run_experiment(
        data_path="/mnt/sdc/xiuxian/chr/new_AD_MCI_AD.npz",
        task=current_task, 
        model_name='CBGT', 
        base_root='./results', 
        batch_size=32,      # 论文中样本量较小，BatchSize 不宜过大
        lr=1e-4,            # 论文设定
        epochs=200,
        device='cuda'
    )