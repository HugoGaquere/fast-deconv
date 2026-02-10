#pragma once

#include "cublas_v2.h"
#include "fast_deconv/util/cublas_macros.hpp"
#include "fast_deconv/util/cuda_macros.hpp"

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstdlib>

namespace fast_deconv::core {

class stream_resources {
 public:
  stream_resources() : owns_stream_(true)
  {
    CHECK_CUDA(cudaStreamCreate(&this->stream));
    CHECK_CUBLAS(cublasCreate(&this->cublas_handle));
    CHECK_CUBLAS(cublasSetStream(this->cublas_handle, this->stream));
  };

  /// Construct from an external CUDA stream (e.g., from CuPy).
  /// The caller retains ownership of the stream and must ensure it outlives this object.
  explicit stream_resources(cudaStream_t external_stream)
    : stream(external_stream), owns_stream_(false)
  {
    CHECK_CUBLAS(cublasCreate(&this->cublas_handle));
    CHECK_CUBLAS(cublasSetStream(this->cublas_handle, this->stream));
  };

  stream_resources(const stream_resources&)            = delete;
  stream_resources(stream_resources&&)                 = delete;
  stream_resources& operator=(const stream_resources&) = delete;
  stream_resources& operator=(stream_resources&&)      = delete;

  ~stream_resources()
  {
    if (this->owns_stream_) {
      // We own the stream, so we can safely free async and sync
      if (this->device_workspace != nullptr) {
        CHECK_CUDA(cudaFreeAsync(this->device_workspace, this->stream));
      }
      CHECK_CUDA(cudaStreamSynchronize(this->stream));
      CHECK_CUBLAS(cublasDestroy(this->cublas_handle));
      CHECK_CUDA(cudaStreamDestroy(this->stream));
    } else {
      // Borrowed stream - the owner (e.g., CuPy) may have already destroyed
      // the stream/context during Python GC, so we must be defensive.
      // Don't use async operations or CHECK macros that would abort on error.
      if (this->device_workspace != nullptr) { cudaFree(this->device_workspace); }
      cublasDestroy(this->cublas_handle);
    }
  };

  void alloc_device(size_t bytes)
  {
    if (bytes == 0) { return; }
    if (this->device_workspace_size >= bytes) { return; }

    if (this->device_workspace != nullptr) {
      CHECK_CUDA(cudaFreeAsync(this->device_workspace, this->stream));
      this->device_workspace = nullptr;
    }

    CHECK_CUDA(cudaMallocAsync(static_cast<void**>(&this->device_workspace), bytes, this->stream));
    this->device_workspace_size = bytes;
  }

  void sync() { CHECK_CUDA(cudaStreamSynchronize(stream)); }

  cudaStream_t stream;
  cublasHandle_t cublas_handle;
  size_t device_workspace_size = 0;
  void* device_workspace       = nullptr;

 private:
  bool owns_stream_;
};

}  // namespace fast_deconv::core
