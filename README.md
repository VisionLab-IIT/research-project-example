# Research Project Example
Example to show possible approaches for project organization, config handling and experiment tracking for deep learning research.

## Preparations
To use the example project, you should first make the necessary preparations to clone it and install the necessary packages.

### 0. Clone the repository
Open a terminal and navigate to the directory where you would like the project to be placed, then
```bash
git clone https://github.com/VisionLab-IIT/research-project-example.git
```
### 1. Change your shell's working directory to the project
```bash
cd research-project-example
```

### 2. (optional) Create a Python virtual environment
It is good practice to keep installed packages of separate projects in their own virtual environments. On Linux, you can create a Python virtual environment with
```bash
python3 -m venv .venv
```
Here, `.venv` will be the directory of your virtual environment. 

### 3. Install requirements
You can install the requirements from `requirements.txt`.

#### 3.1. If you want to install in your virtual environment, activate it first
```bash
source .venv/bin/activate
```
#### 3.2. Then install the requirements
```bash
pip install -r requirements.txt
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

## Training
> [!IMPORTANT]
> Training instructions may differ between each stage, so please read carefully!

### 1. If you work in virtual environment, activate it.
Before running the training script, activate the environment with
```bash
source .venv/bin/activate
```

### 2. Download Dataset
The currently used dataset is [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html). You can download it to the project directory with the following scipt:
```bash
python3 scripts/download_dataset.py --data_path=./data
```
where `--data_path` will be the location of the download.

### 3. Start Training

To train the example model (which is inspired by the [ConvNeXt](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf) architecture), run
```bash
python3 main_training.py --data_path=./data --config_path=config/train.yaml
```
> [!NOTE]
> OmegaConf can support type safety with structured configs which is not presented in this stage.

> [!TIP]
> You can override YAML config parameters with CLI parameters like this (overriding lr):
> ```bash
> python3 main_training.py --data_path=./data --config_path=config/train.yaml optimizer.params.lr=1e-4
> ```
