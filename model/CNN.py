# -*- coding: utf-8 -*-
"""
Created on Sun May 25 17:10:56 2025

@author: ChenPiQ
"""
import torch
import torch.nn.functional as F

class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)


class BrainNetCNN(torch.nn.Module):
    def __init__(self, roi_num=100):
        super().__init__()
        self.in_planes = 1
        self.d = roi_num

        self.e2econv1 = E2EBlock(1, 32, roi_num, bias=True)
        self.e2econv2 = E2EBlock(32, 64, roi_num, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, 2)

    def forward(self,
                node_feature: torch.tensor):
        node_feature =   node_feature[...,0]
        node_feature = node_feature.unsqueeze(dim=1)
        out = F.leaky_relu(self.e2econv1(node_feature), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(
            self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(
            self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(
            self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out
