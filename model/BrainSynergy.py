import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# ===================================================================
# 1. 基础组件：Encoder 与 MLP Expert
# ===================================================================

class BrainEncoder(nn.Module):
    def __init__(self, node_num=100, depth=2, heads=4, dim_feedforward=1024):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=node_num, nhead=heads, 
                dim_feedforward=dim_feedforward, batch_first=True
            ) for _ in range(depth)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim=1) 

class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.GELU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

# ===================================================================
# 2. 交互专家包装器 (处理 Mask 逻辑)
# ===================================================================

class InteractionExpertWrapper(nn.Module):
    def __init__(self, expert_model):
        super(InteractionExpertWrapper, self).__init__()
        self.expert_model = expert_model

    def _forward_with_replacement(self, inputs, replace_index=None):
        processed_inputs = list(inputs)
        if replace_index is not None:
            # 用随机高斯噪声替换指定模态，模拟信息缺失
            random_vector = torch.randn_like(processed_inputs[replace_index])
            processed_inputs[replace_index] = random_vector
        
        x = torch.cat(processed_inputs, dim=-1)
        return self.expert_model(x)    

    def forward_multiple(self, inputs):
        """
        返回一个列表：[原始输出, 替换模态0后的输出, 替换模态1后的输出]
        """
        outputs = []
        outputs.append(self._forward_with_replacement(inputs, replace_index=None))
        for i in range(len(inputs)):
            outputs.append(self._forward_with_replacement(inputs, replace_index=i))
        return outputs

# ===================================================================
# 3. 核心：脑网络信息协同模块 (BrainSynergyModule)
# ===================================================================

class BrainSynergyModule(nn.Module):
    def __init__(self, node_num=100, hidden_dim=256, lambdas=None):
        super().__init__()
        self.num_modalities = 2 
        
        # 1. 特征提取器
        self.encoder_s = BrainEncoder(node_num=node_num)
        self.encoder_f = BrainEncoder(node_num=node_num)
        
        # 2. 实例化四类专家
        # 每类专家都通过 Wrapper 包装，以便调用 forward_multiple
        expert_in = node_num * 2
        self.expert_uni_s = InteractionExpertWrapper(Expert(expert_in, hidden_dim, hidden_dim*2))
        self.expert_uni_f = InteractionExpertWrapper(Expert(expert_in, hidden_dim, hidden_dim*2))
        self.expert_syn = InteractionExpertWrapper(Expert(expert_in, hidden_dim, hidden_dim*2))
        self.expert_rdn = InteractionExpertWrapper(Expert(expert_in, hidden_dim, hidden_dim*2))

        # 损失权重
        if lambdas is None:
            self.lambdas = {'uni_s': 1.0, 'uni_f': 1.0, 'syn': 1.0, 'rdn': 0.1}
        else:
            self.lambdas = lambdas

    # --- 损失函数定义 ---
    def uniqueness_loss(self, anchor, pos, neg):
        # Triplet Loss: 鼓励 anchor 靠近 pos (完关键模态输入)，远离 neg (缺失关键模态)
        triplet_loss = nn.TripletMarginLoss(margin=0.3, p=2, eps=1e-7)
        return triplet_loss(anchor, pos, neg)

    def synergy_loss(self, anchor, negatives):
        """
        协同性：完整输入提取的信息 (anchor) 应该与任何单一模态 (negatives) 提取的信息不同
        """
        total_syn_loss = 0
        anchor_normalized = F.normalize(anchor, p=2, dim=-1)
        for negative in negatives:
            negative_normalized = F.normalize(negative, p=2, dim=-1)
            cosine_sim = torch.sum(anchor_normalized * negative_normalized, dim=1)
            total_syn_loss += torch.mean(cosine_sim)
        return total_syn_loss / len(negatives)

    def redundancy_loss(self, anchor, positives):
        """
        冗余性：完整输入提取的信息 (anchor) 应该在缺失单一模态后依然能被保持 (positives)。
        """
        total_red_loss = 0
        anchor_normalized = F.normalize(anchor, p=2, dim=-1)
        for positive in positives:
            positive_normalized = F.normalize(positive, p=2, dim=-1)
            cosine_sim = torch.sum(anchor_normalized * positive_normalized, dim=1)
            total_red_loss += torch.mean(1 - cosine_sim)
        
        return total_red_loss / len(positives)

    def forward(self, x):
        # x: [B, 100, 100, 2]
        feat_f = self.encoder_f(x[..., 0]) 
        feat_s = self.encoder_s(x[..., 1]) 
        inputs = [feat_f, feat_s] 

        # 获取各专家的输出列表 [All, Mask_0(即Mask_f), Mask_1(即Mask_s)]
        out_uni_s = self.expert_uni_s.forward_multiple(inputs)
        out_uni_f = self.expert_uni_f.forward_multiple(inputs)
        out_syn = self.expert_syn.forward_multiple(inputs)
        out_rdn = self.expert_rdn.forward_multiple(inputs)

        # --- 修改点 2: 调整 Uniqueness Loss 逻辑 ---
        
        # uni_s (结构独特性) 应该依赖模态1 (feat_s)
        # 负样本是缺失模态1 (Mask_1)，正样本是缺失模态0 (Mask_0)
        # triplet_loss(anchor, positive, negative)
        loss_uni_s = self.uniqueness_loss(out_uni_s[0], out_uni_s[1], out_uni_s[2])
        
        # uni_f (功能独特性) 应该依赖模态0 (feat_f)
        # 负样本是缺失模态0 (Mask_0)，正样本是缺失模态1 (Mask_1)
        loss_uni_f = self.uniqueness_loss(out_uni_f[0], out_uni_f[2], out_uni_f[1])

        # --- Synergy 和 Redundancy Loss 逻辑不变 ---
        # 因为它们对 Mask_0 和 Mask_1 是对称处理的（遍历了所有缺失情况）
        loss_syn = self.synergy_loss(out_syn[0], out_syn[1:])
        loss_rdn = self.redundancy_loss(out_rdn[0], out_rdn[1:])

        # 总损失
        total_loss = self.lambdas['uni_s'] * loss_uni_s + \
                     self.lambdas['uni_f'] * loss_uni_f + \
                     self.lambdas['syn'] * loss_syn + \
                     self.lambdas['rdn'] * loss_rdn

        return {
            "loss": total_loss,
            "loss_details": {
                "uni_s": loss_uni_s,
                "uni_f": loss_uni_f,
                "syn": loss_syn,
                "rdn": loss_rdn
            },
            "embeddings": {
                "uni_s": out_uni_s[0], "uni_f": out_uni_f[0],
                "syn": out_syn[0], "rdn": out_rdn[0]
            }
        }

class BrainSynergyClassifier(nn.Module):
    def __init__(self, synergy_module, hidden_dim=256, num_classes=2, dropout=0.5):
        super(BrainSynergyClassifier, self).__init__()
        
        self.synergy_module = synergy_module
        
       
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
            nn.Softmax(dim=-1) 
        )
        
       
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x: [B, 100, 100, 2] 原始脑网络数据
        """
       
        synergy_res = self.synergy_module(x)
        embeddings = synergy_res["embeddings"]
        
       
        expert_list = [embeddings[k] for k in ["uni_s", "uni_f", "syn", "rdn"]]
        combined_features = torch.cat(expert_list, dim=-1) 
        

        weights = self.gate(combined_features) 
        

        stacked_experts = torch.stack(expert_list, dim=1) 
        

        expanded_weights = weights.unsqueeze(-1)

        fused_embedding = torch.sum(stacked_experts * expanded_weights, dim=1)
        

        logits = self.classifier(fused_embedding)
 
        return {
            "logits": logits,
            "weights": weights,               
            "synergy_loss": synergy_res["loss"] 
        }