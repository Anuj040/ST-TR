# The based unit of graph convolutional networks.

from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Variable


class unit_gcn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A,
        use_local_bn: bool = False,
        kernel_size: int = 1,
        stride: int = 1,
        mask_learning: bool = False,
    ):
        super().__init__()

        # ==========================================
        # number of nodes
        self.V = A.size()[-1]

        # the adjacency matrixes of the graph
        self.A = Variable(A.clone(), requires_grad=False).view(-1, self.V, self.V)

        # number of input channels
        self.in_channels = in_channels

        # number of output channels
        self.out_channels = out_channels

        # if true, use mask matrix to reweight the adjacency matrix
        self.mask_learning = mask_learning

        # number of adjacency matrix (number of partitions)
        self.num_A = self.A.size()[0]

        # if true, each node have specific parameters of batch normalizaion layer.
        # if false, all nodes share parameters.
        self.use_local_bn = use_local_bn
        # ==========================================

        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=(kernel_size, 1),
                    padding=(int((kernel_size - 1) / 2), 0),
                    stride=(stride, 1),
                    bias=False,
                )
                for _ in range(self.num_A)
            ]
        )
        if mask_learning:
            self.mask = nn.Parameter(torch.ones(self.A.size()))
        if use_local_bn:
            self.bn = nn.BatchNorm1d(self.out_channels * self.V)
        else:
            self.bn = nn.BatchNorm2d(self.out_channels)

        self.relu = nn.ReLU()

    def forward(
        self, x: torch.tensor, space_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        N, C, T, V = x.size()
        A = self.A.to(x.get_device()).to(x.dtype)

        # reweight adjacency matrix
        if self.mask_learning:
            A = A * self.mask

        if space_mask is not None:
            space_mask = space_mask.unsqueeze(1).tile(1, A.size(0), 1, 1)
            A = A.unsqueeze(0) * (1 - space_mask)
            # graph convolution
            for i, a in enumerate(A.permute(1, 0, 2, 3)):
                xa = torch.matmul(x.reshape(N, C * T, V), a).reshape(N, C, T, V)
                y = self.conv_list[i](xa) if i == 0 else y + self.conv_list[i](xa)
        else:
            # graph convolution
            for i, a in enumerate(A):
                xa = x.reshape(-1, V).mm(a).reshape(N, C, T, V)
                y = self.conv_list[i](xa) if i == 0 else y + self.conv_list[i](xa)

        # batch normalization
        if self.use_local_bn:
            y = y.permute(0, 1, 3, 2).contiguous().view(N, self.out_channels * V, T)
            y = self.bn(y)
            y = y.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)
        else:
            y = self.bn(y)

        # nonliner
        y = self.relu(y)

        return y
