# RLOps Policies Incubator

The **rl_incubator** project provides a modular, scalable framework for training reinforcement learning (RL) policies. It integrates various tools such as TorchRL, Metaflow, and Ray to support both local development and cloud-based experiments (e.g., AWS EKS). This repository includes training scripts, configuration files, and utilities for running distributed RL experiments and exporting models to ONNX.

---

## Table of Contents

- [Installation](#installation)
  - [Conda Environment Setup](#conda-environment-setup)
  - [Pip Dependencies](#pip-dependencies)
  - [Optional Dependencies](#optional-dependencies)
- [Usage](#usage)
  - [Running Locally](#running-locally)
  - [Running with `torchrun`](#running-with-torchrun)
  - [AWS & Kubernetes Configuration](#aws--kubernetes-configuration)
  - [Metaflow Runs](#metaflow-runs)
- [Additional Information](#additional-information)
- [TODO](#todo)

---

## Installation

### Conda Environment Setup

Create and configure your Conda environment with the following commands:

```bash
conda create --name rlops_env python=3.11
conda install -y -c conda-forge glew
conda install -y -c anaconda mesa-libgl-cos6-x86_64
conda install -y -c menpo glfw3
conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
conda deactivate && conda activate rlops_env
```

### Pip Dependencies

Install the required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include:
```
torch
tensordict
gymnasium==0.29.1
pygame
numpy==1.26.4
matplotlib
tqdm
torchrl
mujoco
imageio
metaflow
tensorboard
wandb
```

### Optional Dependencies

For additional functionality, install these optional packages:

```bash
pip install ray[default]==2.8.1
pip install onnx-pytorch
pip install onnxruntime
pip install metaflow-ray
```

---

## Usage

### Running Locally

To run a baseline training locally, use the following command:

```bash
python train_ppo.py --config rl_incubator/configs/experiment_config_baseline.yaml
```

### Running with `torchrun`

For distributed training using `torchrun`, execute:

```bash
torchrun --nnodes=1 --nproc-per-node=1 --max-restarts=1 --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 train_ppo.py --config rl_incubator/configs/experiment_config_baseline.yaml
```

Or to run a tuned configuration:

```bash
torchrun --nnodes=1 --nproc-per-node=1 --max-restarts=1 --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 train_ppo.py --config rl_incubator/configs/experiment_config_tuned_PPO.yaml
```

### AWS & Kubernetes Configuration

- **AWS EKS:**

  Update your kubeconfig for your EKS cluster:

  ```bash
  aws eks update-kubeconfig --name <cluster name>
  ```

- **AWS API Gateway:**

  Retrieve your API key value:

  ```bash
  aws apigateway get-api-key --api-key <YOUR_KEY_ID_FROM_CFN> --include-value | grep value
  ```

- **Metaflow AWS Configuration:**

  Configure Metaflow to use AWS:

  ```bash
  metaflow configure aws
  ```

### Metaflow Runs

Run your training flows on Kubernetes with Metaflow:

```bash
python torch_flow.py --no-pylint --environment=conda run --with kubernetes
```

Other experimental Metaflow runs:

```bash
python ppo_metaflow.py run
python ppo_metaflow.py run --with kubernetes
python ray_flow.py --no-pylint --environment=pypi run --with kubernetes
```

### Tracking Running Processes

If you're using Argo Workflows, you can track the running processes via port forwarding:

```bash
kubectl port-forward -n argo service/argo-argo-workflows-server 2746:2746
```

---

## Additional Information

- **Docker:**  
  Consider using a Dockerfile with a `requirements.txt` to containerize your environment. See the provided Dockerfile in the repository for reference.

- **Model Export:**  
  You can export trained Torch policies to ONNX for deployment on real-world systems.

- **Experiment Tracking:**  
  Integration with W&B and TensorBoard is supported for monitoring training progress.

---

## TODO

- [ ] Set up Training Dependencies with a Docker registry.
- [ ] Use W&B logger for improved experiment tracking.
- [ ] Experiment with Ray Tune on the TorchRL Trainer.
- [ ] Run Metaflow training comparisons.

---
