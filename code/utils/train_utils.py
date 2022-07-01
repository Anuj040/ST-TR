import os

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


def adjust_learning_rate(arg, optimizer, epoch):
    if arg.optimizer not in ["SGD", "Adam", "Adamod"]:
        raise ValueError()
    step = arg.step
    lr = arg.base_lr * (arg.base_lr ** np.sum(epoch >= np.array(step)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(path, filename, state_dict: dict) -> None:
    os.makedirs(path, exist_ok=True)

    try:
        torch.save(
            dict,
            os.path.join(path, filename),
        )

    except Exception as e:
        print("An error occurred while saving the checkpoint:")
        print(e)


def opt_update(optimizer: Optimizer, model: nn.Module, clip_norm: float) -> None:
    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    optimizer.zero_grad()
