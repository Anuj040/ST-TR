# The based unit of graph convolutional networks.

import math
from typing import Optional

import torch
import torch.nn as nn

"""
This class implements Adaptive Graph Convolution. 
Function adapted from "Two-Stream Adaptive Graph Convolutional Networks 
for Skeleton Action Recognition" of Shi. et al. ("https://github.com/lshiwjx/2s-AGCN")

"""


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class UnitAGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A,
        coff_embedding: int = 4,
        num_subset: int = 3,
        use_local_bn: bool = False,
        mask_learning: bool = False,
    ):
        super().__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(A.clone())
        self.A = A
        nn.init.constant_(self.PA, 1e-6)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for _ in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1, bias=False))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1, bias=False))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1, bias=False))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            mask (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        N, C, T, V = x.size()
        A = self.A.to(self.PA.get_device()) + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = (
                self.conv_a[i](x)
                .permute(0, 3, 1, 2)
                .contiguous()
                .view(N, V, self.inter_c * T)
            )
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            if mask is not None:
                mask = mask * -1e9
                A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1) + mask)  # N V V
            else:
                A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))
            A1 = A1 + A[i].to(A1.dtype)
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y) + self.down(x)
        return self.relu(y)
