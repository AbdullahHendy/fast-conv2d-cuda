# fast-conv2d-cuda
**Custom CUDA 2D convolution engines, *FastConv2d*, for a custom CNN model, integrated into PyTorch via C++ extensions, benchmarked against cuDNN on Fashion-MNIST.**

**🎉🥳 Beat PyTorch/cuDNN by [add correct number]x on Nvidia V100 GPU 🥳🎉**

## Details
- Was developed specifically for a [V100 16GB GPU](https://www.nvidia.com/en-gb/data-center/tesla-v100/) but can be run on later architectures with an ***expected*** hit to the **speedup** numbers. 

- The custom Conv2d **replaces nn.Conv2d** in the following model:
    ```python
    model: nn.Module = nn.Sequential(
        nn.Conv2d(1, 12, kernel_size=7, bias=False),  # Or FastConv2d 
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(12, 24, kernel_size=7, bias=False),  # Or FastConv2d
        nn.Flatten(),
        nn.Linear(27 * 27 * 24, 160),
        nn.Tanh(),
        nn.Linear(160, 10),
    )
    ```

- For Conv layer 1, a ***Tiled Convulation*** kernel is used.
- For Conv layer 2, an ***Implicit GEMM Unrolling Convolution*** kernel is used **(Uses Tensor Cores)**.
- For more justification on the **thought process** behind this see [the final report](./report/final-report.pdf). 

- Tested on **Fashion-MNIST datashet** in `data/fashion-mnist/`

- Tested with a **pre-trained checkpoint** in `checkpoints/`
 

## Benchmark Results
### Target:
- **GPU: V100 16GB**
- **CUDA: 11.8**
- **Python: 3.10**
- **PyTorch: 2.0.1+cu118**
>
> | Layer   | PyTorch (ms) | FastConv (ms) | Speedup |
> |:-----:  |:------------:|:------------: |:-------:|
> | Layer 1 | xx.xxxx      | 06.9386       | x.xxx   |
> | Layer 2 | xx.xxxx      | 21.0984       | x.xx×   |
> | **Total Conv Time** | xx.xxxx | **28.0370** | **x.xx×** 

### Extra:
- **GPU: RTX4070 8GB Laptop**
- **CUDA: 13.4**
- **Python: 3.14**
- **PyTorch: 2.14.0+cu132**
>
> | Layer   | PyTorch (ms) | **FastConv (ms)** | Speedup |
> |:-----:  |:------------:|:------------: |:-------:|
> | Layer 1 | 29.1666      | **10.1970**       | 2.86x   |
> | Layer 2 | 22.1819      | **23.4127**       | 0.95×   |
> | **Total Conv Time** | 51.3485 | **33.6097** | **1.53×**

## Report
The [final report](./report/final-report.pdf) discuss in detail the thought process and steps (1to 6) that led to the [final kernels](./csrc/kernels.cuh) used for both convolution layers. The report also discusses **NSight Profiler** report analysis and how they were used to further optimize the kernels.   

## Prerequisites
- **GPU**: NVIDIA GPU with Compute Capability >= 7.0
- **CUDA**: CUDA 11.8+ / 12.x / 13.x with nvcc installed and matching host Nvidia drivers.
- **Build**: CMake >= 3.20, C++20 compiler (GCC >= 11 or Clang >= 13). ***Note**: C++17 supported for legacy PyTorch via CMake flag*.
- **Python**: Python >= 3.9 with Cuda pytorch and Numpy.

> Is that guaranteed to work? of course not!! welcome to Nvidia and python 🙄

---

## Installation
```bash
# Clone project
git clone https://github.com/AbdullahHendy/fast-conv2d-cuda.git
cd fast-conv2d-cuda

# Create python virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu[version]  # Match system version. See: https://pytorch.org/get-started/locally/
pip install numpy

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"

# Build shared library .so needed to hook PyTorch model
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" # Defaults to C++20
# For older PyTorch versions, use C++17
cmake .. \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CUDA_STANDARD=17

cmake --build . -j$(($(nproc) - 2)) # Or whatever -j you like
cd ..
```

## Run Benchmark
### Benchmark script
```bash
python benchmark.py [num_samples]
```
> **SAMPLE SIZE NOTE:** Default evaluates all 10,000 Fashion-MNIST test images in a single batch and all benchmark numbers are using 10000 test images.

> **ACCURACY VERIFICATION NOTE:** The expected accuracy is **79.55%**. This is determined by the pre-trained model weights in `checkpoints/model.pth`.

### Sample Output
```bash
Running benchmark on cuda on a NVIDIA GeForce RTX 4070 Laptop GPU with architecture (8, 9) with 10000 test samples...

Benchmarking PyTorch Native (cuDNN)...
Benchmarking Custom CUDA Engine...

=================================================================
Layer                | PyTorch (ms)   | FastConv (ms)  | Speedup 
-----------------------------------------------------------------
Layer 1              |    29.1666 ms |    10.1970 ms | 2.8603x
Layer 2              |    22.1819 ms |    23.4127 ms | 0.9474x
-----------------------------------------------------------------
Total Conv Time      |    51.3485 ms |    33.6097 ms | 1.5278x
=================================================================
Accuracy -> PyTorch: 79.55% | FastConv: 79.55%
```

## Analysis & Profiling
Use `profile.sh` to run *Nvidia Profiler* analysis.

```bash
# Options
./profile.sh 
# Usage: ./profile.sh [--ncu | --nsys | --memcheck | --racecheck | --synccheck] [--gui]

# Target custom kernels in Nsight Compute (CLI summary)
./profile.sh --ncu

# Run Nsight Compute and immediately open the report in the GUI
./profile.sh --ncu --gui

# Capture timeline trace in Nsight Systems
./profile.sh --nsys --gui

# Memory leak / out-of-bounds check
./profile.sh --memcheck

# Shared memory race condition detection
./profile.sh --racecheck
```
> **NOTE**: If encountered ERR_NVGPUCTRPERM error when running the profiling script, see [this Nvidia Guide](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters) based on your system 
