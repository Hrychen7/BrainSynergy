import os
import random
import numpy as np
import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torch.utils import data

# 导入你放在 model 目录下的模型
from model.BrainSynergy import BrainSynergyModule 

# ================= 1. 数据集类 (支持任务过滤) =================
class ADNIDataset_Pretrain(data.Dataset):
    def __init__(self, npz_path, task='MCI_AD'):
        alldata = np.load(npz_path)
        raw_data = alldata['data']
        raw_labels = alldata['label'].squeeze()

        # 根据任务过滤数据，确保预训练和分类任务的数据分布一致
        if task == 'NC_MCI':
            target_labels = (0, 1)
        elif task == 'MCI_AD':
            target_labels = (1, 2)
        elif task == 'NC_AD':
            target_labels = (0, 2)
        else:
            target_labels = None

        if target_labels is not None:
            mask = (raw_labels == target_labels[0]) | (raw_labels == target_labels[1])
            self.data = raw_data[mask]
            self.labels = raw_labels[mask]
            print(f"[{task}] 预训练模式 | 样本数: {len(self.data)}")
        else:
            self.data = raw_data
            self.labels = raw_labels
            print(f"[All] 全量预训练模式 | 样本数: {len(self.data)}")

        # 自动检测分界点
        diff = np.diff(self.labels)
        split_points = np.where(diff < -0.5)[0]
        self.split_idx = split_points[0] + 1 if len(split_points) > 0 else len(self.labels) // 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ================= 2. 训练/验证 Epoch 逻辑 =================
def run_epoch(model, loader, optimizer, device, is_train=True):
    model.train() if is_train else model.eval()
    metrics = {"total": 0, "uni_s": 0, "uni_f": 0, "syn": 0, "rdn": 0}
    
    with torch.set_grad_enabled(is_train):
        for inputs in loader:
            inputs = inputs.to(device).float()
            out = model(inputs)
            loss = out["loss"]
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics["total"] += loss.item()
            for k in ["uni_s", "uni_f", "syn", "rdn"]:
                metrics[k] += out["loss_details"][k].item()

    for k in metrics: metrics[k] /= len(loader)
    return metrics

# ================= 3. 主实验逻辑 =================
def run_pretrain(args):
    # 固定种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    lambdas = {
        'uni_s': args.w_s,
        'uni_f': args.w_f,
        'syn': args.w_syn,
        'rdn': args.w_rdn
    }
    # 路径设置：./pretrain_results/MCI_AD/BrainSynergy_Pretrain
    setting_str = f"s{args.w_s}_f{args.w_f}_syn{args.w_syn}_rdn{args.w_rdn}"
    exp_dir = os.path.join(args.base_root, args.task, f"Pretrain_{setting_str}")
    os.makedirs(exp_dir, exist_ok=True)
    
    dataset = ADNIDataset_Pretrain(args.data_path, task=args.task)
    indices = np.arange(len(dataset))
    p1_idx, p2_idx = indices[:dataset.split_idx], indices[dataset.split_idx:]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    p1_f, p2_f = list(kf.split(p1_idx)), list(kf.split(p2_idx))

    for fold in range(5):
        print(f"\n>>> Fold {fold} Pretraining for {args.task}...")
        train_idx = np.concatenate([p1_idx[p1_f[fold][0]], p2_idx[p2_f[fold][0]]])
        val_idx = np.concatenate([p1_idx[p1_f[fold][1]], p2_idx[p2_f[fold][1]]])
        
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size)

        model = BrainSynergyModule(node_num=100, hidden_dim=256, lambdas=lambdas).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        fold_dir = os.path.join(exp_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        log_path = os.path.join(fold_dir, "pretrain_detailed_log.txt")
        
        best_val_loss = float('inf')
        with open(log_path, 'w') as f:
            f.write("Epoch,T_Total,T_UniS,T_UniF,T_Syn,T_Rdn,V_Total,V_UniS,V_UniF,V_Syn,V_Rdn\n")

        for epoch in range(args.epochs):
            t_m = run_epoch(model, train_loader, optimizer, args.device, True)
            v_m = run_epoch(model, val_loader, None, args.device, False)
            
            if v_m["total"] < best_val_loss:
                best_val_loss = v_m["total"]
                torch.save(model.state_dict(), os.path.join(fold_dir, "best_pretrain_model.pth"))
                status = "[Best Saved]"
            else: status = ""

            log_line = (f"{epoch+1},{t_m['total']:.6f},{t_m['uni_s']:.6f},{t_m['uni_f']:.6f},"
                        f"{t_m['syn']:.6f},{t_m['rdn']:.6f},{v_m['total']:.6f},"
                        f"{v_m['uni_s']:.6f},{v_m['uni_f']:.6f},{v_m['syn']:.6f},{v_m['rdn']:.6f}\n")
            with open(log_path, 'a') as f: f.write(log_line)
            
            if (epoch + 1) % 10 == 0: 
                print(f"Ep {epoch+1:03d} | ValLoss: {v_m['total']:.4f} {status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrainSynergy Pretraining")
    
    # 核心参数
    parser.add_argument('--task', type=str, default='MCI_AD', choices=['MCI_AD', 'NC_MCI', 'NC_AD'], 
                        help='任务类型，会自动过滤对应的二分类数据')
    parser.add_argument('--data_path', type=str, default='/mnt/sdc/xiuxian/chr/new_AD_MCI_AD.npz')
    parser.add_argument('--base_root', type=str, default='./pretrain_results')
    
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--w_s', type=float, default=1.0, help='Weight for Unique S loss')
    parser.add_argument('--w_f', type=float, default=1.0, help='Weight for Unique F loss')
    parser.add_argument('--w_syn', type=float, default=1.0, help='Weight for Synergy loss')
    parser.add_argument('--w_rdn', type=float, default=1.0, help='Weight for Redundancy loss')

    args = parser.parse_args()
    run_pretrain(args)