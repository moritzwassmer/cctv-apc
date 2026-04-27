# cctv-apc

## Purpose & Overview

This code is designed for training and evaluation of automated passenger counting systems (APC) using CCTV video data. It requires slurm job scheduling and has been tested for the TU Berlin math cluster. It is optimized for parallel grid searches using slurm array jobs.

## Tech Stack

- **PyTorch**: Core deep learning framework for model development and training.
- **PyTorch-Lightning**: Remove ML boilerplate code.
- **Hydra**: Flexible hierarchical configuration composition where parameters may be overriden by command line. Used to launch slurm array jobs to quickly test multiple hyperparameters.
- **MLflow**: Experiment tracking and model lifecycle management. Can be used locally (default) or with a remote tracking server. Lets you browse and manage experiment results via GUI. Lets multiple users share and compare experiment results at the same place.

## Folder Structure

```
cctv-apc/
├── env/                        # conda preperations, folder and link preparation, mlflow setup
├── evaluation/                 # Model evaluation example code
├── napc/                       # Main code modules and configurations 
├── napc/configs/               # Hierarchical configurations (.yaml files)
├── jobs/                       # Job scripts and templates
├── notebooks/                  # Jupyter notebooks for testing, understanding
├── environment.yaml            # Conda environment specification
└── run.py                      # Main entry training or evaluation run 
```

## Installation 

This project assumes conda installed and has been tested on the tu berlin math cluster. It also assumes you have permissions on the used data. Cd into any folder Then follow these steps:

1. **Clone the repository, cd into the root folder:**
    ```bash
    git clone https://github.com/moritzwassmer/cctv-apc
    cd cctv-apc
    ```

2. **Prepare environment:**
    This moves the conda cache to a place where you have write permissions such that you can create a conda environment. Then it creates the conda environment. Then it prepares the file system by creating folders and links. 
    ```bash
    bash env/conda_cache.sh
    bash env/create_env.sh
    bash env/setup_links_paths.sh
    ```

3. **Test if everything works:**
    All jobs assume that the code is run from the main folder of the git repository. also make sure to activate 
    the conda environment before running jobs. 
    ```bash
    conda activate cctv-apc
    bash jobs/examples/test_all_lstm_gpu.sh
    ```
    alternatively use notebooks/run.ipynb to test training in a jupyter notebook

If everything works you will see experiment folders in `exec`, and predictions in `outputs/mlflow/<experiment_id>/artifacts/predictions.csv`

## Usage

a. **Run an experiment:**
    ```bash
    conda activate cctv-apc
    bash jobs/cnn-lstm.sh
    ```

b. **Customized experiment:**

Steps:
- create new configurations in `napc/configs` (if needed) 
- create a job file <your_job>.sh in `jobs` , with optional parameter overwrites (see `jobs` for patterns or see section "Hierarchical Configuration Overview"
- start the run:
```bash
conda activate cctv-apc
bash jobs/<your_job>.sh
```

c. **Evaluation** 

You can find raw csv predictions in `exec/<experiment_name>/out/<run_id>/predictions.csv`
OR in the local mlflow repository `outputs/mlflow/<experiment_id>/<run_id>/artifacts/predictions.csv` 

View the `evaluation` folder for more conveniant ways for evaluation. Use preferably `evaluation/eval_csvs.ipynb` over `evaluation/eval_mlflow.ipynb` as metrics from mlflow sometimes were not perfectly accurate.

## Hierarchical Configuration Overview

To understand the configuration folder structure and syntax i recommend the following documentations:
- Hydra Documentation: https://hydra.cc/docs/advanced/override_grammar/basic/ and https://hydra.cc/docs/advanced/defaults_list/
- OmegaConf Documentation: https://omegaconf.readthedocs.io/en/2.1_branch/index.html

This section provides a short description of the core components of the hierarchical configurations in `napc/configs/`. An example of how a fully composed OmegaConf config may look like can be found in `napc/configs/run/example.yaml`. 

```
napc/configs/
├── run/                        # Glues together all relevant components (model,trainer,datasets)
│   ├── dm/                     # Configures the data module (train test and validation splits)
│   ├── model/                  # Configures the model
│   └── trainer/                # Configures training or inference logic
│       └── callbacks/          # Configures callbacks that should be used during training
└── hydra/                      # Configures hardware parameters (Job runtime, GPU)
```

The program assumes a default configuration (as defined by the default list). For example if you run

```
python run.py
```

That means it will run a dense_lstm_2x64 model on the cctv_vd2_cut2 data module for two epochs (with two batches each epoch). You can override the parameters in the job file as follows:

For example if you want to change a single key value pair like adjusting the learning rate parameter:
```
python run.py --config-name=cpu run.model.lr=0.0001 run.mlflow_creds=/path/to/mlflow_creds/default.yaml
```

when you want to change a whole config group (that is a whole yaml file) such as switching the model from dense_lstm_2x64 to conv_mlstm_2x128, then do the following:
```
python run.py --config-name=cpu  run/model=conv_mlstm_2x128 run.mlflow_creds=/path/to/mlflow_creds/default.yaml
```
change to gpu as follows:
```
python run.py --config-name=gpu_big_math run/model=conv_mlstm_2x128 run.mlflow_creds=/path/to/mlflow_creds/default.yaml
```

## Selected Results

### Metrics
#### Accuracy (ACC):
For this metric, the counting task is interpreted as a sequence classification problem in which each video corresponds to a discrete count label. The classification accuracy is defined as follows, where $\mathbb{1}(\cdot)$ is the indicator function. 

$$
ACC_\theta := \frac{1}{d_K} \sum_{k=1}^{d_K} \mathbb{1}\left(^{(k)}y^\theta = ^{(k)}\hat{y}^\theta\right)
$$

#### Global relative Bias (GRB):
This metric reflects the normalized difference between the total predicted and actual passenger counts across the entire dataset. A negative bias indicates an undercounting bias, whereas a positive bias implies an overcounting bias. The absolute variant $|GRB_\theta|$ avoids favoring results where overcounting and undercounting cancel out, for example, when aggregating the results of multiple seeds. 

$GRB_\theta$ is also used as a main criterion for the approval of APC systems by the Association of German Transport Companies. Ideally, predictions are unbiased, such that:
GRB<sub>θ</sub> ∈ [-1%, 1%]

$$
GRB_\theta := \frac{\sum_{k=1}^{d_K}{} {}^{(k)}\hat{y}^\theta}{\sum_{k=1}^{d_K} {}^{(k)}y^\theta} - 1.
$$

### CNN-LSTM

#### Architecture

The videos are first processed by a CNN that operates over each frame in a video. They are effectively 2D convolutions and progressively downsample the feature maps when being stacked. Note that the temporal dimensions are not downsampled in this model. However I also implemented a model which downsamples the temporal dimensions as well in the encoder and upsamples it back to the original resolution in the decoder (see `jobs/bottleneck_mlstm.sh`)

Then the features are processed by multiple LSTM layers/blocks (either vanilla LSTMs (vLSTM) or xLSTM blocks)

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/217324f5-7be9-4d4d-89a3-4b5459744c0e" />

### Loss Function

Upper and lower bounds are introduced to enable temporal relaxation for the model to not overly punish bad timing. a stand for alighting, b stands for boarding.
<img width="700" height="400" alt="image" src="https://github.com/user-attachments/assets/970cd963-a79d-40ad-87cd-72635a132672" />

The loss is evaluated for every frame in a video. When the model prediction is within the upper and lower bound, the error is zero.  It requires exact event-timing information, hence it requires high precision in annotations.

$$
\mathcal{L}^{\theta}_t := \underbrace{\max(0, \hat{y}^{\theta}_t - u^{\theta}_t)}_{\text{overcount}} - \underbrace{\min(0, \hat{y}^{\theta}_t - l^{\theta}_t)}_{\text{undercount}}
$$

The error for every frame and both heads (a:alighting count, b:boarding count)  is averaged.

$$
\mathcal{L}^* := \frac{1}{2{d_T}} \sum_{t=1}^{d_T} \sum_{\theta\in\{a,b\}} \mathcal{L}^{\theta}_t
$$

#### Results

Accuracy based model selection focuses on achieving the best performance on individual videos.
<img width="600" height="230" alt="image" src="https://github.com/user-attachments/assets/d224d02d-eda4-435c-9d1e-9fe0887ee2e2" />


GRB based model selection focuses on receiving an unbiased (neither overcounting nor undercounting) count over multiple videos.
<img width="600" height="230" alt="image" src="https://github.com/user-attachments/assets/69008b64-01b5-4c19-a724-86d7f9c5d7d2" />

### CNN-vLSTM-256-4 with masked Loss Function

#### Loss Function
This loss does not care about intermediate frames and only cares about the final count of the last frame of a video. It does not require upper or lower bounds and just relies on the original labels. It is easier and possibly cheaper to annotate videos when the exact timing of the events doesn't matter. 

$$
\mathcal{L}^* := \frac{1}{2} \sum_{\theta\in\{a,b\}}MAE(y^\theta_T,\hat{y}^\theta_T)
$$

#### Results

The following are the results of the GRB based model selection. The model has slightly worse $ACC$ (0.3%) than the CNN-vLSTM with the upper and lower bound based loss. However it achieves similar GRB.
<img width="450" height="200" alt="image" src="https://github.com/user-attachments/assets/d6ae3425-e59a-4cd5-aba6-cc33623b28b1" />


## Related Work
This codebase is largely built upon the methodology of:
R. Seidel, N. Jahn, S. Seo, T. Goerttler and K. Obermayer, "NAPC: A Neural Algorithm for Automated Passenger Counting in Public Transport on a Privacy-Friendly Dataset," in IEEE Open Journal of Intelligent Transportation Systems, vol. 3, pp. 33-44, 2022, doi: 10.1109/OJITS.2021.3139393. 
https://ieeexplore.ieee.org/document/9665722

The CNN-LSTM architecture and bounding box extend parameter is adopted from:
S. Seo and K. Obermayer, "CLAPC: A Hybrid CNN-LSTM Architecture for Automated Passenger Counting from Video Streams in Public Transport," in IEEE Open Journal of Intelligent Transportation Systems, 2026, doi: 10.1109/OJITS.2026.3685711.

## Disclaimer
Many docstrings in this repository were generated with the assistance of GitHub Copilot and may contain errors. Please refer to the source code as the authoritative reference for implementation details.
