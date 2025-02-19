from metaflow import FlowSpec, step, current, Flow, resources, conda_base, card
from metaflow.cards import Markdown
# from metaflow.profilers import gpu_profile
import time


# ⬇️ Enable this to test @conda with GPU or on workstations
#@conda_base(
#   python="3.11.0",
#   packages={"pytorch::pytorch-cuda": "12.4", "pytorch::pytorch": "2.4.0"},
#)

# ⬇️ Alternatively, enable this to test @conda on Mac OS X (no CUDA)
@conda_base(python='3.11.0', packages={'pytorch::pytorch': '2.4.0'})

class TorchTestFlow(FlowSpec):
    
    # ⬇️ Enable these two lines to test GPU execution
    # @gpu_profile()
    # @resources(gpu=1, memory=8000)

    @card(type="blank", refresh_interval=1, id="status")
    @step
    def start(self):
        t = self.create_tensor()
        self.run_squarings(t)
        self.next(self.end)

    def create_tensor(self, dim=5000):
        import torch  # pylint: disable=import-error

        print("Creating a random tensor")
        self.tensor = t = torch.rand((dim, dim))
        print("Tensor created! Shape", self.tensor.shape)
        print("Tensor is stored on", self.tensor.device)
        if torch.cuda.is_available():
            print("CUDA available! Moving tensor to GPU memory")
            t = self.tensor.to("cuda")
            print("Tensor is now stored on", t.device)
        else:
            print("CUDA not available")
        return t

    def run_squarings(self, tensor, seconds=60):
        import torch  # pylint: disable=import-error

        print("Starting benchmark")
        counter = Markdown("# Starting to square...")
        current.card["status"].append(counter)
        current.card["status"].refresh()

        count = 0
        s = time.time()
        while time.time() - s < seconds:
            for i in range(25):
                # square the tensor!
                torch.matmul(tensor, tensor)
            count += 25
            counter.update(f"# {count} squarings completed ")
            current.card["status"].refresh()
        elapsed = time.time() - s

        msg = f"⚡ {count/elapsed} squarings per second ⚡"
        current.card["status"].append(Markdown(f"# {msg}"))
        print(msg)

    @step
    def end(self):
        # show that we persisted the tensor artifact
        print("Tensor shape is still", self.tensor.shape)


if __name__ == "__main__":
    TorchTestFlow()