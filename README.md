# Research Project Example
Example to show possible approaches for project organization, config handling and experiment tracking for deep learning research.

## Preparations
To use the example project, you should first make the necessary preparations to clone it and install the necessary packages.

> [!NOTE]
> From stage 10 onwards, [uv](https://docs.astral.sh/uv/#installation) is required for package and project management.

### 0. Clone the repository
Open a terminal and navigate to the directory where you would like the project to be placed, then
```bash
git clone https://github.com/VisionLab-IIT/research-project-example.git
```
### 1. Change your shell's working directory to the project
```bash
cd research-project-example
```


## Checkout the development stage of your choice
You can navigate between the different stages of the project's development. Choose a stage from the list below so you can explore its changes.

Use `git checkout` to switch between stages:
```bash
git checkout <stage>
```
Where `<stage>` can be either the stage tag or the corresponding commit hash.

Here is the list of the most important stages:

| Stage | Tag | Commit | Description |
|-------|---------------------|------|--------|
| 1. Baseline | 01_baseline | 27659e3 | The first state worth checking out as a starting point. |
| 2. Simple Tracker | 02_simple_tracker | 82a1267 | `ExperimentTracker` class to encapsulate functionalities for logging, comparison and reproducibility. <br>At this stage, only the basics from stage 01 are reorganized here. |
| 3. Log Directories | 02_log_dirs | e5b5cee | Tracking logs into separate directories under log. <br>This enables basic comparisons like checking plots of different runs. |
| 4. Using TensorBoard | 04_tensorboard | 6b742dd | Logging into [TensorBoard](https://www.tensorflow.org/tensorboard) for better visualizations and comparison.|
| 5. Basic Config | 05_basic_config | 556feef | - Introducting basic YAML-based configuration for better reproducibility.<br>- Moving dataset download to separate script under helpers. |
| 6. Using OmegaConf | 06_omegaconf | 146971f | Using [OmegaConf](https://omegaconf.readthedocs.io) for convenient config handling and object-style config access. |
| 7. Dynamic Loading | 07_dynamic_loading | 84f347b | Loading model, optimizer, scheduler and loss function dynamically based on config.<br> See the [`getattr()`](https://docs.python.org/3/library/functions.html#getattr) documentation for details. |
| 8. Package Structure | 08_package_structure | 2197bda | - Organizing code into `research_project` Python package for clarity. <br>- Renaming `helpers/` to `scripts/` as it includes a runnable script. <br>- Renaming the repo from `research_project_example` to `research-project-example` in alignment with conventions. |
| 9. Pythonic Refactor | 09_pythonic_refactor | deb1e26 | Refactoring code to follow Python conventions and idioms:<br>- Replacing underscores with hyphens for CLI arguments<br>- Using context manager for training time measurement<br>- Using `torch.no_grad()` decorator instead of context manager for validation function<br>- Introducing type annotations more widely in the project<br>- Using single entrypoint functions for `main_training.py` and `scripts/download_dataset.py`<br>- Replacing `log_scalar` with `log_scalars` for batch logging<br>- Refactoring config.yaml for easier hyperparameter logging<br>- Returning dataclass object in validation function<br>- Cleaning complete progress bars from terminal and adding per-epoch metric logging<br>- Using f-strings and `str.join()` in `metrics_printer`<br>- Organizing config loading and merging into a separate function in `main_training.py`|
| 10. Modern Tooling | 10_modern_tooling |  | Use modern Python project tooling: [Ruff](https://docs.astral.sh/ruff/) for linting and formatting and [uv](https://docs.astral.sh/uv/) for package management. |

## Training
> [!IMPORTANT]
> Training instructions may differ between each stage, so please read carefully!

### 1. Sync environment
This will automatically create a virtual environment and install the exact dependency versions the used in the dev env.
```bash
uv sync
```

### 2. Download Dataset
The currently used dataset is [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html). You can download it to the project directory with the following script:
```bash
uv run scripts/download_dataset.py --data-path=./data
```
where `--data-path` will be the location of the download.

### 3. Start Training

To train the example model (which is inspired by the [ConvNeXt](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf) architecture), run
```bash
uv run main_training.py --data-path=./data --config-path=config/train.yaml
```

> [!TIP]
> You can override YAML config parameters with CLI parameters like this (overriding lr):
> ```bash
> uv run main_training.py --data-path=./data --config-path=config/train.yaml optimizer.params.lr=1e-4
> ```
