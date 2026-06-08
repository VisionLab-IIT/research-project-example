from tqdm import tqdm
import torch
from research_project.utils.metrics import accuracy_metric


def train_one_epoch(
        model,
        dataloader,
        loss_fn,
        optimizer,
        scheduler,
        device,
):
    model.train()

    total_loss = 0.0
    for batch_number, (x, y_true) in tqdm(enumerate(dataloader), total=len(dataloader)):
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

    avg_loss = total_loss.item()/len(dataloader)
    
    return avg_loss


# TODO: Use torch.no_grad() decorator instead of context manager in later stages
def validate_one_epoch(
        model,
        dataloader,
        loss_fn,
        device,
):
    model.eval()

    with torch.no_grad():
        total_acc = 0.0
        total_loss = 0.0
        for batch_number, (x, y_true) in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, y_true = x.to(device), y_true.to(device)

            y_pred = model(x)
            
            loss = loss_fn(y_pred, y_true)
            total_loss += loss.detach()
            
            acc = accuracy_metric(y_pred, y_true)
            total_acc += acc.detach()

    avg_loss = total_loss.item()/len(dataloader)
    avg_acc = total_acc.item()/len(dataloader)
    
    return avg_loss, avg_acc