from typing import Callable

import torch

EPS = 1e-10


def focal_weighted_bce(weights: torch.Tensor, gamma: float = 2.0) -> Callable:

    """
    Multi-label cross-entropy

    Args:
        weights (torch.Tensor): Freq weights for each label class

    Returns:
        Callable: _description_
    """

    def _loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_true (torch.Tensor): true value
            y_pred (torch.Tensor): predicted probabilities

        Returns:
            torch.Tensor: _description_
        """
        first_term = weights * y_true * torch.log(y_pred + EPS) * (1 - y_pred) ** gamma
        second_term = (1 - y_true) * torch.log(1 - y_pred + EPS) * (y_pred) ** gamma
        return -torch.mean(first_term + second_term)

    return _loss
