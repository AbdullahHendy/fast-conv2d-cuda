cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(fast_conv_cuda LANGUAGES CXX CUDA)

# Default to C++20 (required by modern PyTorch), but allow -D overrides for older PyTorch
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")

# Find PyTorch / LibTorch
find_package(Torch REQUIRED)

# Build the shared library directly from our source files
add_library(fast_conv SHARED 
    csrc/binding.cpp 
    csrc/forward.cu
)

target_include_directories(fast_conv PRIVATE csrc)
target_link_libraries(fast_conv "${TORCH_LIBRARIES}" cudart)
