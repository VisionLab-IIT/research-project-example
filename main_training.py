from pathlib import Path
import argparse
from datetime import datetime
import random

import numpy as np
from omegaconf import OmegaConf

import torch
from torch import optim
from torch.optim import lr_scheduler
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize

import research_project.models as models
from research_project.engine_training import train_one_epoch, validate_one_epoch
from research_project.utils.tracker import ExperimentTracker


def get_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument(
        "--config_path", 
        type=Path,
        help="Path to the YAML configuration file",
        required=True
    )
    # Data
    parser.add_argument(
        "--data_path", 
        type=Path, 
        help="Path to the dataset",
        required=True
    )

    args, unknown_args = parser.parse_known_args()
    
    return args, unknown_args


def main(args, config):
    # Basic reproducibility settings
    random.seed(config.seed)  # If Python random is used
    np.random.seed(config.seed)  # If NumPy random is used
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    # Sometimes using deterministic algorithms may be difficult
    #torch.use_deterministic_algorithms(True)

    # Transforms
    train_transforms = Compose(
        [
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
            Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    
    val_transforms = Compose(
        [
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
            Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

    # Datasets
    train_set = CIFAR100(
        root=args.data_path,
        train=True,
        transform=train_transforms,
        target_transform=torch.tensor,
    )

    val_set = CIFAR100(
        root=args.data_path,
        train=False,
        transform=val_transforms,
        target_transform=torch.tensor,
    )

    # Dataloaders
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False
    )

    # Model
    # Finding model class based on config
    # Converting config.model.params to keyword arguments
    model = getattr(models, config.model.name)(**config.model.params)
    model = model.to(config.device)

    # Optimizer, Scheduler, Loss, Tracking
    # Finding optimizer class based on config
    # Converting config.optimizer.params to keyword arguments
    optimizer = getattr(optim, config.optimizer.name)(
        params=model.parameters(),
        **config.optimizer.params,
    )

    # Finding scheduler class based on config
    scheduler_cls = getattr(lr_scheduler, config.scheduler.name)
    scheduler_params = config.scheduler.params
    # Extending sheduler parameters
    if scheduler_cls is lr_scheduler.OneCycleLR:
        scheduler_params["total_steps"] = config.num_epochs*len(train_loader)
    # Converting scheduler_params to keyword arguments
    scheduler = scheduler_cls(
        optimizer=optimizer,
        **scheduler_params
    )

    # Finding loss class based on config
    loss_fn = getattr(torch.nn, config.loss_fn.name)()

    # Main loop
    tracker = ExperimentTracker(
        log_dir=args.log_dir,
        scalar_names=[
            "train_losses",
            "val_losses",
            "val_accs"
        ],
        metric_names=["val_acc"],
        use_tensorboard=True
    )
    tracker.log_training_start()
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scheduler,
            config.device
        )

        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            loss_fn,
            config.device
        )

        is_new_best = tracker.update_metric("val_acc", val_acc)
        if is_new_best:
            torch.save(
                model.state_dict(),
                args.model_weights_path
            )

        tracker.log_scalar(name="train_losses", value=train_loss, index=epoch)
        tracker.log_scalar(name="val_losses", value=val_loss, index=epoch)
        tracker.log_scalar(name="val_accs", value=val_acc, index=epoch)
    tracker.log_training_end()
    
    tracker.log_hparams_and_metrics(
        hparams=dict(
            num_epochs=config.num_epochs,
            lr=config.optimizer.params.lr,
            batch_size=config.batch_size,
            wd=config.optimizer.params.weight_decay
        )
    )
    tracker.finalize_run(
        save_logs=True,
        print_metrics=True
    )
    OmegaConf.save(config, args.log_dir/args.config_path.name)


if __name__ == "__main__":
    # Parse command line arguments
    args, unknown_args = get_args()

    args.data_path = Path(args.data_path)
    args.data_path.mkdir(exist_ok=True)
    
    project_dir = Path(__file__).parent
    # Checkpoint directory
    ckpt_dir = project_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    args.ckpt_dir = ckpt_dir
    args.model_weights_path = args.ckpt_dir / "model.pth"
    
    # Log directory
    run_start_time = datetime.strftime(datetime.now(), "%Y%m%d%H%M")
    log_dir = project_dir / "log" / run_start_time
    log_dir.mkdir(parents=True)
    args.log_dir = log_dir

    # Loading config from YAML
    config = OmegaConf.load(args.config_path)
    # Loading config overrides from CLI
    cli_config = OmegaConf.from_dotlist(unknown_args)
    # Filtering only keys that exist in YAML-based config
    cli_config = {k:v for k, v in cli_config.items() if k in config.keys()}
    cli_config = OmegaConf.create(cli_config)
    # Overriding YAML config with CLI config
    config = OmegaConf.merge(config, cli_config)
    print("---------- Config ----------")
    print(OmegaConf.to_yaml(config).rstrip('\n'))
    print("----------------------------")

    main(args, config)
