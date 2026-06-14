from pathlib import Path
import argparse
from datetime import datetime
import random

import numpy as np
from omegaconf import OmegaConf, DictConfig

import torch
from torch import optim
from torch.optim import lr_scheduler
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize

import research_project.models as models
from research_project.engine_training import train_one_epoch, validate_one_epoch
from research_project.utils.tracker import ExperimentTracker


def get_args() -> tuple["argparse.Namespace", list[str]]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-path", 
        type=Path,
        help="Path to the YAML configuration file",
        required=True
    )
    parser.add_argument(
        "--data-path", 
        type=Path, 
        help="Path to the dataset",
        required=True
    )

    args, unknown_args = parser.parse_known_args()
    
    return args, unknown_args


def load_config(config_path: Path, unknown_args: list[str]) -> DictConfig:
    # Loading config from YAML
    config = OmegaConf.load(config_path)
    
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

    return config


def main():
    # Parse command line arguments
    args, unknown_args = get_args()
    config = load_config(args.config_path, unknown_args)
    
    project_dir = Path(__file__).resolve().parent
    # Checkpoint directory
    ckpt_dir = project_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    
    # Log directory
    run_start_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")
    log_dir = project_dir / "runs" / run_start_time
    log_dir.mkdir(parents=True)

    # Basic reproducibility settings
    random.seed(config.seed)  # If Python random is used
    np.random.seed(config.seed)  # If NumPy random is used
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    # Sometimes using deterministic algorithms may be difficult
    #torch.use_deterministic_algorithms(True)

    # Transforms
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    train_transforms = Compose(
        [
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
            Normalize(mean=rgb_mean, std=rgb_std)
        ]
    )
    
    val_transforms = Compose(
        [
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
            Normalize(mean=rgb_mean, std=rgb_std)
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
        batch_size=config.hparams.batch_size,
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
        scheduler_params["total_steps"] = config.hparams.num_epochs*len(train_loader)
    # Converting scheduler_params to keyword arguments
    scheduler = scheduler_cls(
        optimizer=optimizer,
        **scheduler_params
    )

    # Finding loss class based on config
    loss_fn = getattr(torch.nn, config.loss_fn.name)()

    # Main loop
    tracker = ExperimentTracker(
        log_dir=log_dir,
        scalar_names=[
            "train_losses",
            "val_losses",
            "val_accs"
        ],
        metric_names=["val_acc"],
        use_tensorboard=True
    )
    with tracker.timer("Total training time"):
        for epoch in range(config.hparams.num_epochs):
            train_loss = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                scheduler,
                config.device
            )

            val_res = validate_one_epoch(
                model,
                val_loader,
                loss_fn,
                config.device
            )

            print(
                f"Epoch {epoch} |",
                f"Train loss: {train_loss:.4f} |",
                f"Val loss: {val_res.loss:.4f} |",
                f"Val accuracy: {val_res.accuracy:.4f}"
            )

            if tracker.update_metric("val_acc", val_res.accuracy):
                torch.save(
                    model.state_dict(),
                    ckpt_dir / "model.pth"
                )

            tracker.log_scalars(
                {
                    "train_losses": train_loss,
                    "val_losses": val_res.loss,
                    "val_accs": val_res.accuracy, 
                },
                index=epoch
            )
        
    tracker.log_hparams(OmegaConf.to_container(config.hparams, resolve=True))
    tracker.finalize_run(
        save_logs=True,
        print_metrics=True
    )
    OmegaConf.save(config, log_dir/args.config_path.name)


if __name__ == "__main__":
    main()
