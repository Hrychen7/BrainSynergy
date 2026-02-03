import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from model.TAN import net, MyTriplet_loss
from torch.utils import data
import torch.nn.functional as F
import sys
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
        elif task == 'NC_AD':        # 新增 NC vs AD 任务
            target_labels = (0, 2)
        else:
            raise ValueError("Task must be 'NC_MCI' or 'MCI_AD'")

        diff = np.diff(raw_labels)
        split_points = np.where(diff < -0.5)[0]
        raw_split_idx = split_points[0] + 1 if len(split_points) > 0 else len(raw_labels) // 2

        mask = (raw_labels == target_labels[0]) | (raw_labels == target_labels[1])
        self.data = raw_data[mask]
        
        new_labels = raw_labels[mask]
        final_labels = np.zeros_like(new_labels)
        final_labels[new_labels == target_labels[1]] = 1
        self.labels = final_labels.astype(np.int64)
        self.split_idx = np.sum(mask[:raw_split_idx])
        
        print(f"[{task}] 初始化完成 | 样本总数: {len(self.labels)}")
        print(f"Block 1: {self.split_idx} | Block 2: {len(self.labels)-self.split_idx}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ================= 2. 训练与评估逻辑 =================
def train_epoch(model, criterion1, criterion2, optimizer, loader, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, targets in loader:
        data, targets = data.to(device).float(), targets.to(device)
        data = data.view(data.size(0), data.size(1), -1)
        embedding, _ = model(data)
        predict = model.frozen_forward(data)
        
        loss = criterion1(predict, targets) + criterion2(embedding, targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        total_loss += loss.item()
        correct += (predict.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device).float(), targets.to(device)
            data = data.view(data.size(0), data.size(1), -1)
            predict = model.frozen_forward(data)
            
            probs = torch.softmax(predict, dim=1)[:, 1]
            all_preds.extend(predict.argmax(1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    
    # 指标计算
    acc = np.mean(all_preds == all_targets) * 100
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
    
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(all_targets, all_preds)
    
    try: auc = roc_auc_score(all_targets, all_probs)
    except: auc = 0
    
    return acc, sen, spe, f1, auc

# ================= 3. 主程序 =================
def run_experiment(data_path, task, model_name, base_root, batch_size=64, lr=0.0003, epochs=200, device='cuda', seed=42):
    final_output_dir = os.path.join(base_root, task, model_name)
    os.makedirs(final_output_dir, exist_ok=True)
    
    dataset = ADNIDataset(data_path, task=task)
    indices = np.arange(len(dataset))
    part1_idx = indices[:dataset.split_idx]
    part2_idx = indices[dataset.split_idx:]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p1_folds = list(skf.split(part1_idx, dataset.labels[part1_idx]))
    p2_folds = list(skf.split(part2_idx, dataset.labels[part2_idx]))

    fold_results = [] # 用于存储每一折的最佳指标

    for fold in range(5):
        print(f"\n--- {model_name} | {task} | Fold {fold} ---")
        train_idx = np.concatenate([part1_idx[p1_folds[fold][0]], part2_idx[p2_folds[fold][0]]])
        test_idx = np.concatenate([part1_idx[p1_folds[fold][1]], part2_idx[p2_folds[fold][1]]])
        
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size)

        model = net(roi_num=100, num_classes=2, depth=1, heads=1, dropout=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        c1, c2 = nn.CrossEntropyLoss(), MyTriplet_loss(margin=0.8, loss_weight=0.5)

        fold_dir = os.path.join(final_output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        log_file = os.path.join(fold_dir, "train_log.txt")
        
        best_acc = 0
        best_metrics = None

        for epoch in range(epochs):
            t_loss, t_acc = train_epoch(model, c1, c2, optimizer, train_loader, device)
            v_acc, v_sen, v_spe, v_f1, v_auc = evaluate(model, test_loader, device)
            
            if v_acc > best_acc:
                best_acc = v_acc
                best_metrics = [v_acc, v_sen, v_spe, v_f1, v_auc]
                torch.save(model.state_dict(), os.path.join(fold_dir, f"{model_name}_best.pth"))
            
            with open(log_file, 'a') as f:
                f.write(f"Ep {epoch+1:03d} | Loss: {t_loss:.4f} | TestAcc: {v_acc:.2f}% | F1: {v_f1:.4f}\n")

        fold_results.append(best_metrics)
        print(f"Fold {fold} Best Acc: {best_acc:.2f}%")

    # ================= 4. 计算均值与标准差 =================
    fold_results = np.array(fold_results)
    means = np.mean(fold_results, axis=0)
    stds = np.std(fold_results, axis=0)
    
    summary_path = os.path.join(final_output_dir, "final_summary.txt")
    metrics_names = ["Accuracy", "Sensitivity", "Specificity", "F1-Score", "AUC"]
    
    with open(summary_path, 'w') as f:
        f.write(f"Experiment Summary ({model_name} on {task})\n")
        f.write("="*45 + "\n")
        f.write(f"{'Metric':<15} | {'Mean':<10} | {'Std':<10}\n")
        f.write("-" * 45 + "\n")
        for name, m, s in zip(metrics_names, means, stds):
            unit = "%" if name == "Accuracy" else ""
            f.write(f"{name:<15} | {m:>8.2f}{unit} | {s:>8.2f}{unit}\n")
        f.write("="*45 + "\n")
    
    print(f"\n实验结束！汇总结果 (Mean ± Std) 已保存至: {summary_path}")

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)        # 补充：固定 Python 哈希随机性
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)                    # 补充：当前 GPU 的种子
    torch.cuda.manual_seed_all(seed)                # 所有 GPU 的种子

    # cuDNN 相关设置
    torch.backends.cudnn.deterministic = True       # 确保每次返回的卷积算法一致
    torch.backends.cudnn.benchmark = False          # 补充：禁用自动寻找最快算法，避免波动
    current_task = sys.argv[1] if len(sys.argv) > 1 else 'MCI_AD'
    run_experiment(
        data_path="/mnt/sdc/xiuxian/chr/new_AD_MCI_AD.npz",
        task=current_task, 
        model_name='TAN', 
        base_root='./results', 
        epochs=200,
        device='cuda'
    )