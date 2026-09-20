import torch
import torch.nn as nn
import sys
import os
from torch.utils.data import TensorDataset
from collections import defaultdict
import warnings

from utils.reader import load_mnist

# Load custom CUDA kernel operator
SO_PATH = os.path.join(os.path.dirname(__file__), "build/libfast_conv.so")
torch.ops.load_library(SO_PATH)

os.environ["CUDA_CACHE_DISABLE"] = "1"
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
torch.backends.cudnn.benchmark = True

DATA_DIR = "data/fashion-mnist"
MODEL_PATH = "checkpoints/model.pth"

time_dict = defaultdict(
    lambda: [torch.cuda.Event(enable_timing=True) for _ in range(2)]
)


def load_fashion_mnist(path, dataset_size, device):
    test_images, test_labels = load_mnist(path, rows=72, cols=72, kind="t10k-72")
    images = torch.tensor(test_images, dtype=torch.float32, device=device)
    labels = torch.tensor(test_labels, dtype=torch.float32, device=device)

    dataset = TensorDataset(images, labels)
    return torch.utils.data.Subset(dataset, range(dataset_size))


class FastConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )

    def forward(self, input):
        return torch.ops.fast_conv.forward(input, self.weight, self.out_channels)


def build_model(conv_cls, device):
    """Instantiate CNN architecture using either custom or PyTorch Conv2d."""
    model: nn.Module = nn.Sequential(
        conv_cls(1, 12, kernel_size=7, bias=False),  # FastConv2d or nn.Conv2d
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        conv_cls(12, 24, kernel_size=7, bias=False),  # FastConv2d or nn.Conv2d
        nn.Flatten(),
        nn.Linear(27 * 27 * 24, 160),
        nn.Tanh(),
        nn.Linear(160, 10),
    ).to(device)
    model.eval()

    return model

def benchmark_model(model, test_loader, conv_cls):
    layer_events = []

    def pre_hook(layer, inp):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        layer._start_event = start

    def post_hook(layer, inp, out):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        layer_events.append((layer, layer._start_event, end))

    # Hook only the convolution layers
    for layer in model.children():
        if isinstance(layer, conv_cls):
            layer.register_forward_pre_hook(pre_hook)
            layer.register_forward_hook(post_hook)

    # Warmup pass
    evaluate(model, test_loader)
    layer_events.clear()

    # Timed inference pass
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    accuracy = evaluate(model, test_loader)
    torch.cuda.profiler.stop()
    torch.cuda.synchronize()

    # Extract layer execution times in ms
    conv_times_ms = [start.elapsed_time(end) for _, start, end in layer_events]
    return conv_times_ms, accuracy


@torch.no_grad()
def evaluate(model, test_loader, zero=False):
    correct = total = 0

    for images, labels in test_loader:
        outputs = model(torch.zeros_like(images) if zero else images)
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()

    return correct / total


def main():
    dataset_size = 10000
    # Parse command line arguments
    if len(sys.argv) > 1:
        dataset_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        print("Usage:", sys.argv[0], "<dataset size>")
        print("    <dataset_size> = [0 - 10000]")
        sys.exit(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading fashion-mnist data...")
    test_dataset = load_fashion_mnist(DATA_DIR, dataset_size, device)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=len(test_dataset),  # Use full dataset as one batch
        shuffle=False,
    )
    print("Loading model weights...")
    state_dict = torch.load(MODEL_PATH, weights_only=True)

    print(f"Running benchmark on {device} on a {torch.cuda.get_device_name(device)} with architecture {torch.cuda.get_device_capability()} with {dataset_size} test samples...\n")

    # Benchmark PyTorch Native Baseline (nn.Conv2d / cuDNN)
    print("Benchmarking PyTorch Native (cuDNN)...")
    baseline_model = build_model(nn.Conv2d, device)
    baseline_model.load_state_dict(state_dict)
    base_times, base_acc = benchmark_model(baseline_model, test_loader, nn.Conv2d)

    # Benchmark Custom CUDA Kernels (FastConv2d)
    print("Benchmarking Custom CUDA Engine...")
    custom_model = build_model(FastConv2d, device)
    custom_model.load_state_dict(state_dict)
    cust_times, cust_acc = benchmark_model(custom_model, test_loader, FastConv2d)

    # 3. Print Results & Speedup
    print("\n" + "=" * 65)
    print(f"{'Layer':<20} | {'PyTorch (ms)':<14} | {'FastConv (ms)':<14} | {'Speedup':<8}")
    print("-" * 65)
    for i, (b_t, c_t) in enumerate(zip(base_times, cust_times), 1):
        speedup = b_t / c_t if c_t > 0 else 0.0
        print(f"Layer {i:<14} | {b_t:>10.4f} ms | {c_t:>10.4f} ms | {speedup:>6.4f}x")
    print("-" * 65)
    total_b = sum(base_times)
    total_c = sum(cust_times)
    print(f"{'Total Conv Time':<20} | {total_b:>10.4f} ms | {total_c:>10.4f} ms | {total_b/total_c:>6.4f}x")
    print("=" * 65)
    print(f"Accuracy -> PyTorch: {base_acc*100:.2f}% | FastConv: {cust_acc*100:.2f}%\n")


if __name__ == "__main__":
    main()
