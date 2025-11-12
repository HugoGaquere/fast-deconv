#!/bin/bash

cmake -S . -B build/ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc && cmake --build build/
