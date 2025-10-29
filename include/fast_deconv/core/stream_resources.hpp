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
    if (this->device_workspace != nullptr) {
      CHECK_CUDA(cudaFreeAsync(this->device_workspace, this->stream));
      this->device_workspace       = nullptr;
      this->device_workspace_size  = 0;
    }

    if (this->device_output_workspace != nullptr) {
      CHECK_CUDA(cudaFreeAsync(this->device_output_workspace, this->stream));
      this->device_output_workspace      = nullptr;
      this->device_output_workspace_size = 0;
    }

    CHECK_CUDA(cudaStreamSynchronize(this->stream));

    if (this->host_workspace != nullptr) {
      CHECK_CUDA(cudaFreeHost(this->host_workspace));
      this->host_workspace      = nullptr;
      this->host_workspace_size = 0;
    }
    CHECK_CUDA(cudaStreamDestroy(this->stream));
  };

  void alloc_device(size_t bytes, cudaStream_t request_stream = nullptr)
  {
    if (bytes == 0) { return; }

    cudaStream_t stream_to_use = request_stream != nullptr ? request_stream : this->stream;

    if (this->device_workspace_size >= bytes) { return; }

    if (this->device_workspace != nullptr) {
      CHECK_CUDA(cudaFreeAsync(this->device_workspace, stream_to_use));
      this->device_workspace = nullptr;
    }

    CHECK_CUDA(cudaMallocAsync(static_cast<void**>(&this->device_workspace), bytes, stream_to_use));
    this->device_workspace_size = bytes;
  }

  void alloc_device_output(size_t bytes, cudaStream_t request_stream = nullptr)
  {
    if (bytes == 0) { return; }

    cudaStream_t stream_to_use = request_stream != nullptr ? request_stream : this->stream;

    if (this->device_output_workspace_size >= bytes) { return; }

    if (this->device_output_workspace != nullptr) {
      CHECK_CUDA(cudaFreeAsync(this->device_output_workspace, stream_to_use));
      this->device_output_workspace = nullptr;
    }

    CHECK_CUDA(
      cudaMallocAsync(static_cast<void**>(&this->device_output_workspace), bytes, stream_to_use));
    this->device_output_workspace_size = bytes;
  }

  void alloc_host(size_t bytes)
  {
    if (bytes == 0) { return; }

    if (this->host_workspace_size >= bytes) { return; }

    if (this->host_workspace != nullptr) {
      CHECK_CUDA(cudaFreeHost(this->host_workspace));
      this->host_workspace = nullptr;
    }

    CHECK_CUDA(cudaMallocHost(&this->host_workspace, bytes));
    this->host_workspace_size = bytes;
  }

  cudaStream_t stream;
  size_t host_workspace_size            = 0;
  size_t device_workspace_size          = 0;
  size_t device_output_workspace_size   = 0;
  void* host_workspace                  = nullptr;
  void* device_workspace                = nullptr;
  void* device_output_workspace         = nullptr;
};

}  // namespace fast_deconv::core
