



```shell

conda create --name rlops_env python=3.9

pip install torch tensordict  gymnasium==0.29.1 pygame
pip install numpy==1.26.4 
pip install matplotlib
 pip install tqdm
 pip install torchrl
 conda install -c conda-forge glew
conda install -c anaconda mesa-libgl-cos6-x86_64
conda install -c menpo glfw3
conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
conda deactivate && conda activate rlops_env
pip install mujoco
pip install imageio
# pip install ray 
pip install ray[default]==2.8.1 

pip install metaflow


pip install onnx-pytorch
pip install onnxruntime

pip install tensorboard
pip install wandb


# Fault tolerant (fixed sized number of workers, no elasticity, tolerates 3 failures)
torchrun --nnodes=1 --nproc-per-node=1 --max-restarts=1 --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=1 train_ppo.py --config rl_incubator/configs/experiment_config_baseline.yaml


torchrun --nnodes=1 --nproc-per-node=1 --max-restarts=1 --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=1 train_ppo.py --config rl_incubator/configs/experiment_config_tuned_PPO.yaml


```# rl_incubator
