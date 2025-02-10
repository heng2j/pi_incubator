from metaflow import FlowSpec, step
import subprocess

class PPOTrainingFlow(FlowSpec):

    @step
    def start(self):
        print("Starting PPO Training Flow...")
        self.next(self.train_policy)

    @step
    def train_policy(self):
        print("Training PPO policy using TorchRL Trainer...")
        subprocess.run(["python", "train_ppo.py"], check=True)
        self.next(self.convert_to_onnx)

    @step
    def convert_to_onnx(self):
        print("Converting trained policy to ONNX format...")
        # subprocess.run(["python", "convert_to_onnx.py"], check=True)
        self.next(self.end)

    @step
    def end(self):
        print("Metaflow PPO Training Flow completed.")

if __name__ == "__main__":
    PPOTrainingFlow()
