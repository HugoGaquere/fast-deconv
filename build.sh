#!/bin/bash

export MATHDX_ROOT="/usr/local/nvidia-cuda12/mathdx/25.12/"
conan build . -b missing

# conan build -b missing
# cmake --preset conan-release # cmake configure with default build type (release)
# cmake --build --preset conan-release # build the library

 # cmake -S . -B build/ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCUTENSOR_ROOT=/usr && cmake --build build/
