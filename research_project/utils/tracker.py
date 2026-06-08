from pathlib import Path
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter
from research_project.utils.metrics_printer import decorate_metrics


class ExperimentTracker:
    def __init__(
            self,
            log_dir,
            scalar_names, 
            metric_names,
            use_tensorboard=False
    ):
        assert isinstance(scalar_names, (list, tuple)), f"scalar_names must be either list or tupple, got {type(scalar_names)}"
        assert isinstance(metric_names, (list, tuple)), f"metric_names must be either list or tupple, got {type(metric_names)}"

        self.log_dir = Path(log_dir)
        self.scalar_names = scalar_names
        self.metric_names = metric_names
        self.tensorboard_writer = None
        if use_tensorboard:
            self.tensorboard_writer = SummaryWriter(self.log_dir)

        self.scalars = dict()
        for scalar_name in scalar_names:
            self.scalars[scalar_name] = []
        self.best_metrics = dict()
        for metric_name in metric_names:
            self.best_metrics[metric_name] = 0.0
        
    def log_scalar(
            self, 
            name:str, 
            value,
            index
    ):
        self.scalars[name].append(value)
        
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(name, value, index)

    def log_hparams_and_metrics(
        self,
        hparams:dict
    ):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_hparams(
                hparam_dict=hparams,
                metric_dict=self.best_metrics,
                run_name="."
            )

    def update_metric(
            self,
            name:str,
            value
    ):
        result = (value > self.best_metrics[name])
        if result:
            self.best_metrics[name] = value

        return result
    
    def print_best_metrics(self):
        print(decorate_metrics(self.best_metrics))
    
    def log_training_start(self):
        self.train_start_time = datetime.now()

    def log_training_end(self):
        self.train_end_time = datetime.now()
        self.train_duration = self.train_end_time-self.train_start_time
        print(f"Total training time: {self.train_duration}")

    def save_logs(self):
        for scalar_name in self.scalar_names:
            scalar_log_file = self.log_dir / Path(scalar_name+".json")
            with open(scalar_log_file, "w") as f:
                json.dump(self.scalars[scalar_name], f)

        with open(self.log_dir/"metrics.txt", "w") as f:
            f.write(decorate_metrics(self.best_metrics))

    def finalize_run(
            self,
            save_logs:bool=False,
            print_metrics:bool=False
    ):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()

        if save_logs:
            self.save_logs()
        if print_metrics:
            self.print_best_metrics()