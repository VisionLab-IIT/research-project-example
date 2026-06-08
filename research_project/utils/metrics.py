import torch


def accuracy_metric(y_pred, y_true):
    acc = 100*(torch.argmax(y_pred, dim=1)==y_true).float().mean()
    return acc
