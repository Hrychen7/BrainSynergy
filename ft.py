import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import torch.nn.functional as F
import sys
import argparse

# 导入你的模型定义
from model.BrainSynergy import BrainSynergyModule, BrainSynergyClassifier

# ================= 1. 数据集类 (复用你提供的分类逻辑) =================
class ADNIDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, task='NC_MCI'):
        alldata = np.load(npz_path)
        raw_data = alldata['data']
        raw_labels = alldata['label'].squeeze()

        if task == 'NC_MCI':
            target_labels = (0, 1)
        elif task == 'MCI_AD':
            target_labels = (1, 2)
        elif task == 'NC_AD':
            target_labels = (0, 2)
        else:
            raise ValueError("Task error")

        diff = np.diff(raw_labels)
        split_points = np.where(diff < -0.5)[0]
        raw_split_idx = split_points[0] + 1 if len(split_points) > 0 else len(raw_labels) // 2

        mask = (raw_labels == target_labels[0]) | (raw_labels == target_labels[1])
        self.data = raw_data[mask]
        
        filtered_labels = raw_labels[mask]
        final_labels = np.zeros_like(filtered_labels)
        final_labels[filtered_labels == target_labels[1]] = 1
        self.labels = final_labels.astype(np.int64)
        self.split_idx = np.sum(mask[:raw_split_idx])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ================= 2. 训练与评估逻辑 =================
def train_epoch(model, optimizer, loader, device, alpha_syn=0.1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, targets in loader:
        data, targets = data.to(device).float(), targets.to(device)
        
        optimizer.zero_grad()
        out = model(data)
        logits = out["logits"]
        syn_loss = out["synergy_loss"] # 预训练时的解耦约束
        
        # 总损失 = 分类损失 + 协同约束损失 (保持特征不坍塌)
        cls_loss = F.cross_entropy(logits, targets)
        loss = cls_loss + alpha_syn * syn_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device).float(), targets.to(device)
            out = model(data)
            logits = out["logits"]
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    acc = np.mean(all_preds == all_targets) * 100
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(all_targets, all_preds)
    try: auc = roc_auc_score(all_targets, all_probs)
    except: auc = 0
    return acc, sen, spe, f1, auc

# ================= 3. 主实验程序 =================
def run_finetune(args):
    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # 路径设置
    exp_name = f"Finetune_lr{args.lr}_alpha{args.alpha_syn}"
    final_output_dir = os.path.join(args.save_root, args.task, exp_name)
    os.makedirs(final_output_dir, exist_ok=True)
    
    dataset = ADNIDataset(args.data_path, task=args.task)
    indices = np.arange(len(dataset))
    part1_idx = indices[:dataset.split_idx]
    part2_idx = indices[dataset.split_idx:]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    p1_folds = list(skf.split(part1_idx, dataset.labels[part1_idx]))
    p2_folds = list(skf.split(part2_idx, dataset.labels[part2_idx]))

    fold_results = []

    for fold in range(5):
        print(f"\n>>> Fold {fold} | Task: {args.task} | Finetuning...")
        
        train_idx = np.concatenate([part1_idx[p1_folds[fold][0]], part2_idx[p2_folds[fold][0]]])
        test_idx = np.concatenate([part1_idx[p1_folds[fold][1]], part2_idx[p2_folds[fold][1]]])
        
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size)

        # A. 初始化预训练模块并加载权重
        synergy_module = BrainSynergyModule(node_num=100, hidden_dim=256)
        # 根据你的预训练保存路径加载对应的 fold 权重
        pt_path = os.path.join(args.pt_root, args.task, args.pt_setting, f"fold_{fold}", "best_pretrain_model.pth")
        if os.path.exists(pt_path):
            synergy_module.load_state_dict(torch.load(pt_path))
            print(f"Loaded pretrained weights from {pt_path}")
        else:
            print(f"Warning: No pretrained weights found at {pt_path}, training from scratch.")

        model = BrainSynergyClassifier(synergy_module, hidden_dim=256, num_classes=2).to(args.device)

        optimizer = torch.optim.Adam([
            {'params': model.synergy_module.parameters(), 'lr': args.lr * 0.1}, 
            {'params': model.gate.parameters(), 'lr': args.lr},
            {'params': model.classifier.parameters(), 'lr': args.lr}
        ], weight_decay=args.weight_decay)

        fold_dir = os.path.join(final_output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        best_acc, best_res = 0, None

        for epoch in range(args.epochs):
            t_loss, t_acc = train_epoch(model, optimizer, train_loader, args.device, args.alpha_syn)
            v_acc, v_sen, v_spe, v_f1, v_auc = evaluate(model, test_loader, args.device)
            
            if v_acc > best_acc:
                best_acc = v_acc
                best_res = [v_acc, v_sen, v_spe, v_f1, v_auc]
                torch.save(model.state_dict(), os.path.join(fold_dir, "best_finetune_model.pth"))
            
            if (epoch+1) % 20 == 0:
                print(f"Ep {epoch+1:03d} | Loss: {t_loss:.4f} | TestAcc: {v_acc:.2f}% | AUC: {v_auc:.4f}")

        fold_results.append(best_res)

    # 计算最终统计结果
    fold_results = np.array(fold_results)
    means, stds = np.mean(fold_results, axis=0), np.std(fold_results, axis=0)
    print(f"\nFinal Accuracy: {means[0]:.2f} ± {stds[0]:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='MCI_AD')
    parser.add_argument('--data_path', type=str, default="/mnt/sdc/xiuxian/chr/new_AD_MCI_AD.npz")
    parser.add_argument('--pt_root', type=str, default="./pretrain_results", help="预训练结果根目录")
    parser.add_argument('--pt_setting', type=str, default="Pretrain_s1.0_f1.0_syn1.0_rdn1.0", help="预训练的具体参数文件夹名")
    parser.add_argument('--save_root', type=str, default="./finetune_results")
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--alpha_syn', type=float, default=0.1, help="微调时保持解耦损失的权重")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    run_finetune(args)