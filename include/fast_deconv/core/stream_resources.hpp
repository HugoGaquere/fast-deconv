#pragma once

#include "fast_deconv/util/cuda_macros.hpp"

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstdlib>

namespace fast_deconv::core {

class stream_resources {
 public:
  stream_resources() { CHECK_CUDA(cudaStreamCreate(&this->stream)); };

  ~stream_resources()
  {
    CHECK_CUDA(cudaStreamDestroy(this->stream));
    CHECK_CUDA(cudaFree(this->device_workspace));
    free(this->host_workspace);
  };

  void alloc_device(size_t bytes)
  {
    if (this->device_workspace_size < bytes)
      CHECK_CUDA(
        cudaMallocAsync(static_cast<void**>(&this->device_workspace), bytes, this->stream));
  }

  void alloc_host(size_t bytes)
  {
    if (this->host_workspace_size < bytes)
      this->host_workspace = malloc(bytes);
  }

  cudaStream_t stream;
  size_t host_workspace_size   = 0;
  size_t device_workspace_size = 0;
  void* host_workspace         = nullptr;
  void* device_workspace       = nullptr;
};

}  // namespace fast_deconv::core
