from metaflow import FlowSpec, step, conda_base
import subprocess


# ⬇️ Alternatively, enable this to test @conda on Mac OS X (no CUDA)
@conda_base(
    python="3.11.0",
    packages={
        "pytorch::pytorch": "2.4.0",
        "glew": "",
        "mesa-libgl-cos6-x86_64": "",
        "glfw3": "",
    },
    env_vars={
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
    }
)

class PPOTrainingFlow(FlowSpec):

    @step
    def start(self):
        print("Starting PPO Training Flow...")
        self.next(self.train_policy)

    @step
    def train_policy(self):
        print("Training PPO policy using TorchRL Trainer...")
        subprocess.run(["python", "train_ppo.py", "--config", "rl_incubator/configs/experiment_config_baseline.yaml"], check=True)
        self.next(self.end)

    # @step
    # def convert_to_onnx(self):
    #     print("Converting trained policy to ONNX format...")
    #     # subprocess.run(["python", "convert_to_onnx.py"], check=True)
    #     self.next(self.end)

    @step
    def end(self):
        print("Metaflow PPO Training Flow completed.")

if __name__ == "__main__":
    PPOTrainingFlow()
