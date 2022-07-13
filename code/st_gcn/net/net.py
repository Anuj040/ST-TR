from typing import Optional

import torch
import torch.nn as nn


class Unit2D(nn.Module):
    def __init__(
        self, D_in, D_out, kernel_size, stride=1, dim=2, dropout=0, bias=True
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2

        if dim == 2:
            self.conv = nn.Conv2d(
                D_in,
                D_out,
                kernel_size=(kernel_size, 1),
                padding=(pad, 0),
                stride=(stride, 1),
                bias=bias,
            )
        elif dim == 3:
            self.conv = nn.Conv2d(
                D_in,
                D_out,
                kernel_size=(1, kernel_size),
                padding=(0, pad),
                stride=(1, stride),
                bias=bias,
            )
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(D_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(
        self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.dropout(x)
        return self.relu(self.bn(self.conv(x)))
