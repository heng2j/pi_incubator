



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

```# rl_incubator
