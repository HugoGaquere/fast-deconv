#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <vector>

#include "cublas_v2.h"
#include "fast_deconv/util/cublas_macros.hpp"
#include "fast_deconv/util/cuda_macros.hpp"

namespace fast_deconv::core {

class stream_resources {
 public:
  stream_resources(uint flag)
  {
    CHECK_CUDA(cudaStreamCreateWithFlags(&cuda_stream, flag));
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, cuda_stream));
  }

  stream_resources(stream_resources&&) = delete;
  stream_resources& operator=(stream_resources&&) = delete;
  stream_resources(const stream_resources&) = delete;
  stream_resources& operator=(const stream_resources&) = delete;

  ~stream_resources()
  {
    CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaStreamDestroy(cuda_stream));
  }

  void sync() const { CHECK_CUDA(cudaStreamSynchronize(cuda_stream)); }

  cudaStream_t cuda_stream;
  cublasHandle_t cublas_handle;
};

class stream_resources_pool {
 public:
  static constexpr uint8_t default_size{16};
  static constexpr uint8_t default_flag{cudaStreamDefault};

  explicit stream_resources_pool(uint8_t pool_size = default_size, uint8_t flag = default_flag)
  {
    for (uint8_t i = 0; i < pool_size; i++)
      streams_.push_back(std::make_unique<stream_resources>(flag));
  }

  stream_resources_pool(const stream_resources_pool&) = delete;
  stream_resources_pool& operator=(const stream_resources_pool&) = delete;
  stream_resources_pool(stream_resources_pool&&) = delete;
  stream_resources_pool& operator=(stream_resources_pool&&) = delete;

  const stream_resources& get_stream_ref() const noexcept
  {
    return *streams_[next_stream_.fetch_add(1, std::memory_order_relaxed) % streams_.size()];
  }

  uint8_t size() const noexcept { return streams_.size(); }

 private:
  std::vector<std::unique_ptr<stream_resources>> streams_;
  mutable std::atomic_uint8_t next_stream_{};
};

class resources {
 public:
  resources(uint8_t device) : device_(device)
  {
    cudaMemPoolProps pool_props = {};
    pool_props.allocType = cudaMemAllocationTypePinned;    // page-locked GPU memory
    pool_props.handleTypes = cudaMemHandleTypeNone;        // no inter-process sharing
    pool_props.location.type = cudaMemLocationTypeDevice;  // memory lives on the GPU
    pool_props.location.id = device;                       // GPU to allocate on
    CHECK_CUDA(cudaMemPoolCreate(&memory_pool_, &pool_props));

    // configure memory pool to never release back to os
    uint64_t threshold = UINT64_MAX;
    CHECK_CUDA(cudaMemPoolSetAttribute(memory_pool_, cudaMemPoolAttrReleaseThreshold, &threshold));
  }

  resources(const resources&) = delete;
  resources& operator=(const resources&) = delete;
  resources(resources&&) = delete;
  resources& operator=(resources&&) = delete;

  ~resources() { CHECK_CUDA(cudaMemPoolDestroy(memory_pool_)); }

  const stream_resources& get_stream_resources() const noexcept
  {
    return stream_res_pool_.get_stream_ref();
  }

  template <typename T = void>
  T* alloc_async(uint64_t n, const stream_resources& stream_r) const
  {
    if (n == 0) throw std::invalid_argument("alloc_async: n must be > 0");

    uint64_t num_bytes;
    if constexpr (std::is_void_v<T>) {
      num_bytes = n;
    } else {
      num_bytes = n * sizeof(T);
    }

    T* ptr = nullptr;
    CHECK_CUDA(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&ptr), num_bytes, memory_pool_,
                                       stream_r.cuda_stream));
    return ptr;
  }

  template <typename T>
  void free_async(T* ptr, const stream_resources& stream_r) const
  {
    if (ptr == nullptr) return;
    CHECK_CUDA(cudaFreeAsync(ptr, stream_r.cuda_stream));
  }

 private:
  uint8_t device_;
  cudaMemPool_t memory_pool_;
  stream_resources_pool stream_res_pool_;
};

}  // namespace fast_deconv::core
