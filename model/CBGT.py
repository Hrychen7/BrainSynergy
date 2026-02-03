import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange

# 1. 基础 FeedForward 层
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.net(x) + x)

# 2. Cross-Modal Attention (论文 Section 2.3 Equation 4)
class CrossModalAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x: [B, N, D] (功能特征)
        mask: [B, N, N] (结构掩码 M)
        """
        b, n, d, h = *x.shape, self.heads
        q = self.to_q(x).view(b, n, h, -1).transpose(1, 2)
        k = self.to_k(x).view(b, n, h, -1).transpose(1, 2)
        v = self.to_v(x).view(b, n, h, -1).transpose(1, 2)

        # 核心：计算注意力并应用结构掩码 (1 + M)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # 论文 Eq.4: softmax( (QK/sqrt(d)) * (1 + M) )
        # 注意：mask 需要扩展到 multi-head 维度
        m = mask.unsqueeze(1) # [B, 1, N, N]
        dots = dots * (1 + m)
        
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 3. Cross-Modal TopK Pooling (论文 Section 2.3 Equation 6-8)
class CrossModalTopKPooling(nn.Module):
    def __init__(self, n_roi, dim, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.k = None # 动态计算
        # 映射向量 w_s 和 w_f
        self.w_s = nn.Parameter(torch.randn(n_roi, 1))
        self.w_f = nn.Parameter(torch.randn(dim, 1))
        # 评分 MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        batch, n, d = x.shape
        k = int(self.ratio * n)

        # 1. 结构重要性 s1 = ||M * w_s||
        s1 = torch.norm(torch.matmul(mask, self.w_s[:n]), p=2, dim=-1) # [B, N]
        # 2. 功能重要性 s2 = ||X * w_f||
        s2 = torch.norm(torch.matmul(x, self.w_f), p=2, dim=-1) # [B, N]
        
        # 3. 综合评分
        S = torch.stack([s1, s2], dim=-1) # [B, N, 2]
        scores = self.score_mlp(S).squeeze(-1) # [B, N]

        # 4. TopK 选择
        _, indices = torch.topk(scores, k, dim=-1)
        indices, _ = torch.sort(indices, dim=-1) # 保持顺序

        # 5. 筛选节点和更新掩码 (Eq. 8)
        batch_indices = torch.arange(batch).unsqueeze(-1).expand(-1, k).to(x.device)
        
        # 筛选节点特征并进行加权
        x_pooled = x[batch_indices, indices]
        score_weights = scores[batch_indices, indices].unsqueeze(-1)
        x_pooled = x_pooled * score_weights

        # 更新掩码 M^l = M^{l-1}_{i,i}
        # 这里模拟索引操作: mask[indices, indices]
        mask_pooled = []
        for b in range(batch):
            idx = indices[b]
            m_b = mask[b][idx, :][:, idx]
            mask_pooled.append(m_b)
        mask_pooled = torch.stack(mask_pooled)

        return x_pooled, mask_pooled, indices

# 4. CBGT 主模型
class CBGTNet(nn.Module):
    def __init__(self, roi_num=90, num_classes=2, depth=2, heads=4, dropout=0.5, pooling_ratio=0.7):
        super().__init__()
        self.roi_num = roi_num
        self.depth = depth
        dim = roi_num # 初始节点特征维度

        # 结构特征初步处理 (Equation 1 & 2 的逻辑在 forward 中)
        # 模拟 XGBoost 特征选择的权重层
        self.sc_feature_selector = nn.Parameter(torch.ones(roi_num, roi_num))
        
        # 多层 Transformer + Pooling
        self.layers = nn.ModuleList()
        curr_rois = roi_num
        for i in range(depth):
            self.layers.append(nn.ModuleDict({
                'attention': CrossModalAttention(dim, heads, dim // heads, dropout),
                'norm': nn.LayerNorm(dim),
                'ffn': FeedForward(dim, dim * 2, dropout),
                'pooling': CrossModalTopKPooling(curr_rois, dim, pooling_ratio)
            }))
            curr_rois = int(curr_rois * pooling_ratio)

        # 每一层的分类头 (用于 Soft Voting)
        self.readouts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 128),
                nn.LeakyReLU(),
                nn.Linear(128, num_classes)
            ) for _ in range(depth)
        ])

    def forward(self, inputs):
        # 1. 拆分数据
        # 假设输入格式: [B, N, N*2], 前半部分 FC, 后半部分 SC
        fc = inputs[:, :, :self.roi_num] 
        sc = inputs[:, :, self.roi_num:]

        # 2. 结构特征预处理 (Eq. 1 & 2)
        sc = torch.log10(sc + 1.0)
        sc_mean = sc.mean(dim=(1, 2), keepdim=True)
        sc_std = sc.std(dim=(1, 2), keepdim=True) + 1e-6
        sc = (sc - sc_mean) / sc_std
        
        # 初始增强掩码 M0 (模拟 XGBoost 筛选)
        M = sc * torch.sigmoid(self.sc_feature_selector)
        X = fc # 初始节点特征使用 FC 的行

        layer_probs = []

        # 3. 逐层处理
        for i in range(self.depth):
            # Transformer 块
            X = self.layers[i]['attention'](X, M) + X
            X = self.layers[i]['ffn'](X)
            
            # 保存该层特征用于决策
            # Graph Readout: 对所有剩余节点取平均 (也可以用 Flatten)
            graph_feat = X.mean(dim=1) 
            logits = self.readouts[i](graph_feat)
            layer_probs.append(F.softmax(logits, dim=-1))

            # Pooling 块 (减小图规模)
            if i < self.depth - 1: # 最后一层后通常不需要再池化
                X, M, _ = self.layers[i]['pooling'](X, M)

        # 4. Soft Voting (Eq. 9)
        final_prob = torch.stack(layer_probs, dim=0).mean(dim=0)
        
        # 为了适配您的训练代码，返回 embedding 和最终概率
        # 这里 embedding 取最后一层均值
        return F.normalize(X.mean(dim=1), p=2, dim=1), final_prob

    def frozen_forward(self, x):
        _, probs = self.forward(x)
        return probs