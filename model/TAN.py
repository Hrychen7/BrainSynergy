import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange


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
        residual = x
        x = self.net(x)
        x = self.norm(x + residual)
        return x


class MyAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super(MyAttention, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_Q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_K = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_V = nn.Linear(dim, dim_head * heads, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        residual = x
        b, n_x, d_x, h_x = *x.shape, self.heads
        b, n_y, d_y, h_y = *y.shape, self.heads

        q = self.to_Q(x).view(b, -1, self.heads, d_y // self.heads).transpose(1, 2)
        k = self.to_K(y).view(b, -1, self.heads, d_x // self.heads).transpose(1, 2)
        v = self.to_V(y).view(b, -1, self.heads, d_x // self.heads).transpose(1, 2)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.norm(out + residual)
        return out


class MyCrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super(MyCrossAttention, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_Q = nn.Linear(dim, dim, bias=False)
        self.to_K = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_V = nn.Linear(dim, dim_head * heads, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        residual = y
        b, n_x, d_x, h_x = *x.shape, self.heads
        b, n_y, d_y, h_y = *y.shape, self.heads

        q = self.to_Q(x).view(b, -1, 1, d_x).transpose(1, 2)
        q = q.repeat(1, self.heads, 1, 1)
        k = self.to_K(y).view(b, -1, self.heads, d_y // self.heads).transpose(1, 2)
        v = self.to_V(y).view(b, -1, self.heads, d_y // self.heads).transpose(1, 2)

        kkt = einsum('b h i d, b h j d -> b h i j', k, k) * self.scale
        dots = einsum('b h i d, b h j d -> b h i j', q, kkt) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.norm(out + residual)
        return out


class MyDecoderLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.5):
        super(MyDecoderLayer, self).__init__()
        self.SelfAttention = MyAttention(dim, heads=1, dim_head=dim)
        self.CrossAttention = MyCrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.FeedForward = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, enc_out):
        out = self.SelfAttention(x, x)
        out = self.CrossAttention(out, enc_out)
        out = self.FeedForward(out)
        return out


class MyDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super(MyDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [MyDecoderLayer(dim, heads, dim_head, mlp_dim, dropout=dropout) for _ in range(depth)])

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return x


class MyEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout):
        super(MyEncoderLayer, self).__init__()
        self.SelfAttention = MyAttention(dim, heads, dim_head)
        self.norm = nn.LayerNorm(dim)
        self.FeedForward = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.SelfAttention(x, x)
        x = self.FeedForward(x)
        return x


class MyEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super(MyEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [MyEncoderLayer(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class net(nn.Module):
    def __init__(self, roi_num, num_classes, depth, heads, dropout):
        super(net, self).__init__()
        self.dim = roi_num
        self.rois = roi_num
        mlp_dim = self.dim * 3

        self.encoder = MyEncoder(self.dim, depth, heads, self.dim // heads, mlp_dim, dropout)
        self.decoder = MyDecoder(self.dim, depth, heads, self.dim // heads, mlp_dim, dropout)

        self.to_latent = nn.Identity()
        # 动态计算 fc1 第一层线性层的输入维度
        fc1_in_dim = (roi_num // 2) * (roi_num // 2)
        self.fc1 = nn.Sequential(
            nn.Linear(fc1_in_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        self.norm = nn.LayerNorm(self.rois)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=(2, 2), padding=(0, 0))

    def forward(self, inputs):
        
        mri = inputs[:, :, : -self.rois]
        dti = inputs[:, :, -self.rois:]

        x = self.encoder(mri)
        x_out = torch.matmul(x, x.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.rois))

        y = self.decoder(dti, x)
        y_out = torch.matmul(y, y.transpose(-1, -2) / torch.sqrt(torch.tensor(self.rois)))

        out = x_out + y_out
        out = self.maxpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)

        out_norm = F.normalize(out, p=2, dim=1)
        return out_norm, out

    def frozen_forward(self, x):
        with torch.no_grad():
            _, x = self.forward(x)
        x = self.mlp_head(x)
        return torch.softmax(x, dim=-1)




class MyTriplet_loss(nn.Module):
    def __init__(self, margin=1.0, loss_weight=1.0):
        super(MyTriplet_loss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        # distances
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-16).sqrt()
        ap_dist = dist.unsqueeze(2)
        an_dist = dist.unsqueeze(1)
        triplet_loss = ap_dist - an_dist + self.margin
        # triplets mask
        mask = get_mask(targets)
        # loss
        triplet_loss = torch.multiply(mask, triplet_loss)
        triplet_loss = torch.maximum(triplet_loss, torch.tensor(0.0))
        num_triplets = torch.sum((triplet_loss > 1e-16).float())
        triplet_loss = torch.sum(triplet_loss) / (num_triplets + 1e-16)

        return triplet_loss


def get_mask(targets):
    indices = ~torch.eye(targets.shape[0], dtype=torch.bool, device=targets.device)

    i_j = indices.unsqueeze(2)
    i_k = indices.unsqueeze(1)
    j_k = indices.unsqueeze(0)

    dist_indices = torch.logical_and(torch.logical_and(i_j, i_k), j_k)

    targets_equal = targets.unsqueeze(0).eq(targets.unsqueeze(1))
    i_equal_j = targets_equal.unsqueeze(2)
    i_equal_k = targets_equal.unsqueeze(1)
    valid_labels = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

    mask = torch.logical_and(valid_labels, dist_indices)

    return mask.float()

import torch



def valid(train_loader, valid_loader, model, num_classes):
    predict, y_true = None, None
    trained_embedding, trained_label = None, None

    model.eval()
    with torch.no_grad():
        for i, (inputs1, imputs2,label) in enumerate(train_loader):
            inputs= torch.cat((inputs1,imputs2),dim = 2)
            inputs = inputs.float().cuda()
            label = label.cuda()

            train_embedding, _ = model(inputs)
            train_label = label.view(-1)
            if i == 0:
                trained_embedding = train_embedding
                trained_label = train_label
            else:
                trained_embedding = torch.vstack((trained_embedding, train_embedding))
                trained_label = torch.hstack((trained_label, train_label))

        for i, (inputs1, imputs2,label) in enumerate(valid_loader):
            inputs= torch.cat((inputs1,imputs2),dim = 2)
            inputs = inputs.float().cuda()
            label = label.cuda()

            embedding, _ = model(inputs)

            for j, emb in enumerate(embedding):
                dist = torch.sum((emb - trained_embedding) ** 2, dim=-1) ** 0.5
                top_k = torch.argsort(dist)[:5]
                count = [0 for i in range(num_classes)]
                for k in trained_label[top_k]:
                    count[k] += 1
                emb_label = torch.argmax(torch.tensor(count))

                if i == 0 and j == 0:
                    predict = emb_label
                else:
                    predict = torch.hstack((predict, emb_label))

            label = label.view(-1)

            if i == 0:
                pre = predict
                y_true = label
            else:
                pre = torch.hstack((pre, predict))
                y_true = torch.hstack((y_true, label))

        valid_acc, valid_sen, valid_spe, valid_auc = statistics(y_true.cpu(), pre.cpu())

        return valid_acc, valid_sen, valid_spe, valid_auc


import torch
from sklearn.metrics import accuracy_score, roc_auc_score



def statistics(y_true, pre):
    acc, auc, sen, spe = 0.0, 0.0, 0.0, 0.0
    try:
        ACC = accuracy_score(y_true, pre)
        AUC = roc_auc_score(y_true, pre)
        TP = torch.sum(y_true & pre)
        TN = len(y_true) - torch.sum(y_true | pre)
        true_sum = torch.sum(y_true)
        neg_sum = len(y_true) - true_sum
        SEN = TP / true_sum
        SPE = TN / neg_sum

        acc += ACC
        sen += SEN.cpu().numpy()
        spe += SPE.cpu().numpy()
        auc += AUC

    except ValueError as ve:
        print(ve)
        pass

    return acc, sen, spe, auc