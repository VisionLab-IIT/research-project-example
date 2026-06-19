import torch


def accuracy_metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    acc = 100 * (torch.argmax(y_pred, dim=1) == y_true).float().mean()
    return acc
