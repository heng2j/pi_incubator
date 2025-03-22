# RL Incubator

A modern reinforcement learning framework built with Ray and TorchRL, following best practices and modern Python project structure.

## Features

- Distributed training with Ray
- PPO implementation with TorchRL
- Modern project structure following Python best practices
- Comprehensive testing and development tools
- Configurable training pipelines

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rl_incubator.git
cd rl_incubator

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e ".[dev]"
```

## Project Structure

```
rl_incubator/
├── src/
│   └── rl_incubator/
│       ├── models/         # Neural network models
│       ├── environments/   # RL environments
│       ├── training/       # Training scripts and utilities
│       ├── utils/         # Utility functions
│       └── configs/       # Configuration files
├── tests/                 # Test files
├── notebooks/            # Jupyter notebooks
├── scripts/             # Utility scripts
├── configs/             # Configuration files
├── docs/               # Documentation
└── pyproject.toml      # Project metadata and dependencies
```

## Usage

```python
from rl_incubator.training import PPOTrainer
from rl_incubator.environments import GymEnv
from rl_incubator.models import PPOActorCritic

# Create environment
env = GymEnv("Ant-v4")

# Create model
model = PPOActorCritic(env.observation_space, env.action_space)

# Create trainer
trainer = PPOTrainer(env, model)

# Train the model
trainer.train()
```

## Development

### Running Tests

```bash
pytest
```

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

To format code:
```bash
black .
isort .
```

To run type checking:
```bash
mypy .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# RLOps Policies Incubator


**pi_incubator** is an unified RL training framework designed with RLOps in mind to empower researchers with a modular, scalable framework for training reinforcement learning (RL) policies. It integrates various tools such as TorchRL, Metaflow, and Ray to support both local development and cloud-based experiments (e.g., AWS EKS). This repository includes training scripts, configuration files, and utilities for orchestrating distributed RL experiments at scale, providing frictionless, scalable, and user-friendly infrastructure that accelerates research development cycles


![High-Level User Workflow Diagram](docs/images/user_workflow.png)


---

## Table of Contents

- [High-Level Architecture Diagram](#high-level-architecture-diagram)
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
- [TODOs](#todos)

---

## High-Level Architecture Diagram

![High-Level Architecture Diagram](docs/images/system_diagram.png)

<details open>
<summary>Core Components</summary>
<br>

### External RL Open-Source Repos

- **Sim Env Zoo**: Collection of simulated training environments from various high fidelity simulators (MuJoCo, NVIDIA Isaac etc.)
- **Sim Model Zoo**: Reusable simulation/dynamics models and configurations from the shared simulators for standardized scenarios
- **NN Model Zoo**: Neural architectures tailored for RL (CNNs, RNN, Transformers, Mamba )
- **Policy Algorithm Zoo**: Range of RL algorithms (PPO, SAC, Dreamer, GRPO, etc.)

### Distributed Trainer (TorchRL)

- Leverages PyTorch and TorchRL for training with GPU acceleration

### Distributed Training Data Collection (Ray)

- Parallel rollout workers collecting experiences from high fidelity simulators
- Dynamically orchestrates resource allocation and training workflow using Ray and Ray Tune

### Distributed Replay Buffer (Ray + VectorDB)

- Stores large-scale offline and real-world gathered experience data
- Facilitates off-policy training, data re-sampling, and memory-based RL

### Experiment Orchestrator (Metaflow)

- Defines, executes, and tracks end-to-end training workflows (data prep, training, evaluation, deployment)
- Manages job submission to Kubernetes for auto-scaling

### Inference & Evaluation (Custom with Metaflow + Ray Tune + MLflow)

- Custom inference pipelines for batch evaluation with custom metrics
- Integrates with Ray Tune for hyperparameter tuning, metric aggregation and push to MLflow
- Automated alerts and early stopping to automatically halt underperforming experiments or trigger adjustments based on defined performance metrics.

### Experiment Tracking with Observability (MLflow)

- Stores artifacts (checkpoints, logs, metrics) for reproducibility and comparison
- Provides a dashboard for historical experiment insights

### Optional Policy Distillation Process

- Distills large policies into smaller, efficient models for real-world deployment

### Policy Exporter (Torch → ONNX)

- Converts PyTorch RL policies into ONNX format for edge deployment on embedded devices (Nvidia Jetson, etc.)

### Shared RL Training Optimizers Repos

- Training Optimization method like Population-Based Training (PBT), Auto-curriculum, Domain Randomization (DR), Unsupervised Environment Design (UED), Self-Play

### Custom Experiment Configurations

- User defined experiment configurations (YAML/JSON) handling: Training workflow, environment parameters, hyperparameters, compute resources, deployment settings, etc.
</details>


By leveraging TorchRL, Ray, and Metaflow, they jointly provide an end-to-end RLOps solution—from flexible algorithm design to scalable, reproducible production experiments.**

**[TorchRL](https://github.com/pytorch/rl)**  
- **PyTorch-first:** Leverages PyTorch's flexibility for custom RL algorithms.  
- **Modularity:** Allows rapid prototyping with composable network components.

**[Ray & Ray Tune](https://github.com/ray-project/ray)**  
- **Scalable Rollouts:** Distributes high-fidelity simulation data collection seamlessly.  
- **Automated Tuning:** Integrates hyperparameter search with third-party logging (W&B, MLflow).  
- **Efficient Scaling:** Transitions effortlessly from local experiments to multi-node clusters.

**[Metaflow](https://github.com/Netflix/metaflow)**  
- **Orchestration:** Simplifies complex workflow management and experiment tracking.  
- **Kubernetes Ready:** Automates containerized training jobs for on-prem or cloud deployments.




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


### Optional Dependencies

For additional functionality, install these optional packages:

```bash
pip install ray[default]==2.8.1
pip install onnx-pytorch
pip install onnxruntime
pip install metaflow-ray
```


### Install

To install `pi_incubator`:

```bash
pip install -e .
```


---





## Current Experimental Usage

### Running Locally

To run a baseline training locally, use the following command:

```bash
python train_ppo.py --config configs/experiment_config_baseline.yaml
```

```bash
python torchrl_ray_train.py --config configs/experiment_config_baseline.yaml
```


### Running with `torchrun`

For distributed training using `torchrun`, execute:

```bash
torchrun --nnodes=1 --nproc-per-node=1 --max-restarts=1 --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 train_ppo.py --config configs/experiment_config_baseline.yaml
```

Or to run a tuned configuration:

```bash
torchrun --nnodes=1 --nproc-per-node=1 --max-restarts=1 --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 train_ppo.py --config configs/experiment_config_tuned_PPO.yaml
```



### AWS & Kubernetes Configuration

- **Deploying to AWS with Kubernetes:**

As of now, please go to `terraform-aws-metaflow` dir then follow the official [Outerbounds instruction](https://docs.outerbounds.com/engineering/deployment/aws-k8s/deployment/#apply-terraform-template-to-provision-aws-infrastructure) to deploy these services in your AWS account:AWS EKS cluster, Amazon S3, AWS Fargate and Relational Database Service (RDS). 


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


### Experimental Runs

Investigating why collector kwargs are not being passed to desinated collector from RayCollector. Need to understand how the argument are being pass. Will reachout to TorchRL team.
```bash
python ray_collector_train.py
```

Currently not able to train with RayCollector and TorchRL trainer together. However, by fixing the the following on the official [ray_train.py](https://github.com/pytorch/rl/blob/main/examples/distributed/collectors/multi_nodes/ray_train.py) example. It is working without TorchRL trainer.


`RuntimeError: Setting 'advantage' via the constructor is deprecated, use .set_keys(<key>='some_key') instead.`

Uppon fixed the outdated ClipPPOLoss 

`RuntimeError: The distribution TanhNormal has not analytical mode. Use ExplorationMode.DETERMINISTIC to get a deterministic sample from it.`

Need to set_exploration_type from ExplorationType.MODE to ExplorationType.DETERMINISTIC



---

## TODOs
- [x] Experiment with Ray Tune on the TorchRL Trainer.
- [x] Experiment with Ray Collector with TorchRL Trainer.
- [x] Investigate how collector kwargs from Ray Collector are passing into SyncDataCollector
- [ ] Investigating [Isaac Gym Wrapper](https://pytorch.org/rl/0.6/_modules/torchrl/envs/libs/isaacgym.html#IsaacGymWrapper)
- [ ] Investigating on Self-play env set up
- [ ] Support local Kubernetes with [S3Mock](https://github.com/adobe/S3Mock)
- [ ] Set up Training Dependencies with a Docker registry.
- [ ] Experiment with Ray Collector with Ray Tune + TorchRL Trainer.
- [ ] Use W&B logger for improved experiment tracking.
- [ ] Run Metaflow training comparisons on TunedAdam.


---
