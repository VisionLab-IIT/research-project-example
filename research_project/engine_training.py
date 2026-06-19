from dataclasses import dataclass
from tqdm import tqdm
import torch
from research_project.utils.metrics import accuracy_metric


@dataclass(frozen=True)
class ValidationResult:
    loss: float
    accuracy: float


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: callable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str | torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)

        optimizer.zero_grad()

        # Forward
        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)
        # Backward
        loss.backward()

        # Optimization & Scheduling
        optimizer.step()
        scheduler.step()

        detached_loss = loss.detach()

        total_loss += detached_loss

    avg_loss = total_loss.item() / len(dataloader)

    return avg_loss


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: callable,
    device: str | torch.device,
) -> ValidationResult:
    model.eval()

    total_acc = 0.0
    total_loss = 0.0
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)

        y_pred = model(x)

        loss = loss_fn(y_pred, y_true)
        total_loss += loss.detach()

        acc = accuracy_metric(y_pred, y_true)
        total_acc += acc.detach()

    avg_loss = total_loss.item() / len(dataloader)
    avg_acc = total_acc.item() / len(dataloader)

    return ValidationResult(loss=avg_loss, accuracy=avg_acc)
