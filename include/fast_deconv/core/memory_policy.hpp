#pragma once
#include <cuda_runtime.h>

#include <fast_deconv/util/cuda_macros.hpp>

template <typename T>
struct DeviceMemory {
  static void allocate(T*& ptr, std::size_t n)
  {
    if (n == 0) {
      ptr = nullptr;
      return;
    }
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
  }
  static void free(T* ptr)
  {
    if (ptr) CHECK_CUDA(cudaFree(ptr));
  }
};

template <typename T>
struct ManagedMemory {
  static void allocate(T*& ptr, std::size_t n)
  {
    if (n == 0) {
      ptr = nullptr;
      return;
    }
    CHECK_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
  }
  static void free(T* ptr)
  {
    if (ptr) CHECK_CUDA(cudaFree(ptr));
  }
};
