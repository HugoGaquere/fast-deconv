#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstdio>
#include <cstdlib>
#include <emu/cuda/device/mdcontainer.hpp>
#include <emu/cuda/memory.hpp>
#include <emu/cuda/stream.hpp>
#include <vector>

#include "cublas_v2.h"
#include "fast_deconv/util/cublas_macros.hpp"
#include "fast_deconv/util/cuda_macros.hpp"

namespace fast_deconv::core {

class stream_resources {
 public:
  static constexpr uint default_flag{cudaStreamDefault};

  /// Bind the calling thread to @p device before creating the stream and
  /// cuBLAS handle, so both end up on the requested GPU regardless of which
  /// device the thread had active before.
  stream_resources(const cudaMemPool_t& mem_pool, uint flag, uint8_t device) : device(device), mem_pool_(mem_pool)
  {
    CHECK_CUDA(cudaSetDevice(device));
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

  template <typename T = void>
  T* alloc_async(uint64_t n) const
  {
    if (n == 0) throw std::invalid_argument("alloc_async: n must be > 0");

    uint64_t num_bytes;
    if constexpr (std::is_void_v<T>)
      num_bytes = n;
    else
      num_bytes = n * sizeof(T);

    T* ptr = nullptr;
    CHECK_CUDA(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&ptr), num_bytes, mem_pool_, cuda_stream));
    return ptr;
  }

  template <typename T>
  void free_async(T* ptr) const
  {
    if (ptr == nullptr) return;
    CHECK_CUDA(cudaFreeAsync(ptr, cuda_stream));
  }

  void sync() const { CHECK_CUDA(cudaStreamSynchronize(cuda_stream)); }

  void switch_to_device() const { CHECK_CUDA(cudaSetDevice(device)); };

  cudaStream_t cuda_stream;
  cublasHandle_t cublas_handle;
  uint8_t device;

 private:
  const cudaMemPool_t& mem_pool_;
};

class resources {
 public:
  explicit resources(uint8_t device) : device_(device), memory_pool_(make_memory_pool(device)) {}

  resources(const resources&) = delete;
  resources& operator=(const resources&) = delete;
  resources(resources&&) = delete;
  resources& operator=(resources&&) = delete;

  ~resources() { CHECK_CUDA(cudaMemPoolDestroy(memory_pool_)); }

  const cudaMemPool_t& mem_pool() const noexcept { return memory_pool_; }

  /// Mint a stream (+ cuBLAS handle) on this device, allocating from this
  /// pool. The returned stream_resources references this object's mem pool,
  /// so it must not outlive these resources. Callers own the stream they
  /// create — there is no shared pool of streams to alias with.
  stream_resources make_stream(uint flag = stream_resources::default_flag) const
  {
    return stream_resources(memory_pool_, flag, device_);
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
    CHECK_CUDA(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&ptr), num_bytes, memory_pool_, stream_r.cuda_stream));
    return ptr;
  }

  template <typename T>
  void free_async(T* ptr, const stream_resources& stream_r) const
  {
    if (ptr == nullptr) return;
    CHECK_CUDA(cudaFreeAsync(ptr, stream_r.cuda_stream));
  }

  /// Bytes currently allocated by user code (live working set).
  uint64_t pool_used_bytes() const
  {
    cuuint64_t v = 0;
    CHECK_CUDA(cudaMemPoolGetAttribute(memory_pool_, cudaMemPoolAttrUsedMemCurrent, &v));
    return static_cast<uint64_t>(v);
  }

  /// Peak working set since the pool was created (or last reset).
  uint64_t pool_used_peak_bytes() const
  {
    cuuint64_t v = 0;
    CHECK_CUDA(cudaMemPoolGetAttribute(memory_pool_, cudaMemPoolAttrUsedMemHigh, &v));
    return static_cast<uint64_t>(v);
  }

  /// Bytes the pool currently holds from the driver (used + retained).
  uint64_t pool_reserved_bytes() const
  {
    cuuint64_t v = 0;
    CHECK_CUDA(cudaMemPoolGetAttribute(memory_pool_, cudaMemPoolAttrReservedMemCurrent, &v));
    return static_cast<uint64_t>(v);
  }

  /// Peak pool reservation since creation.
  uint64_t pool_reserved_peak_bytes() const
  {
    cuuint64_t v = 0;
    CHECK_CUDA(cudaMemPoolGetAttribute(memory_pool_, cudaMemPoolAttrReservedMemHigh, &v));
    return static_cast<uint64_t>(v);
  }

  /// Reset the pool's high-water marks (UsedMemHigh, ReservedMemHigh) to current.
  void pool_reset_peaks() const
  {
    cuuint64_t zero = 0;
    CHECK_CUDA(cudaMemPoolSetAttribute(memory_pool_, cudaMemPoolAttrUsedMemHigh, &zero));
    CHECK_CUDA(cudaMemPoolSetAttribute(memory_pool_, cudaMemPoolAttrReservedMemHigh, &zero));
  }

  /// Device-wide free / total memory (independent of the pool).
  void device_mem_info(size_t& free_b, size_t& total_b) const
  {
    CHECK_CUDA(cudaSetDevice(device_));
    CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
  }

  /// Print pool + device memory state to stderr. @p label is prepended for grep'ing.
  void print_memory_usage(const char* label = "") const
  {
    size_t free_b = 0, total_b = 0;
    device_mem_info(free_b, total_b);
    fprintf(stderr,
            "[mem%s%s] used=%.2f GB  reserved=%.2f GB  (peak used=%.2f GB  peak reserved=%.2f GB)  "
            "device-free=%.2f / %.2f GB\n",
            (*label) ? " " : "", label, pool_used_bytes() / 1e9, pool_reserved_bytes() / 1e9,
            pool_used_peak_bytes() / 1e9, pool_reserved_peak_bytes() / 1e9, free_b / 1e9, total_b / 1e9);
  }

 private:
  /// Build the device mem pool before any stream member could reference it —
  /// members initialize in declaration order, so the pool handle must be a
  /// fully-created value by the time anything downstream binds to it.
  static cudaMemPool_t make_memory_pool(uint8_t device)
  {
    cudaMemPoolProps pool_props = {};
    pool_props.allocType = cudaMemAllocationTypePinned;    // page-locked GPU memory
    pool_props.handleTypes = cudaMemHandleTypeNone;        // no inter-process sharing
    pool_props.location.type = cudaMemLocationTypeDevice;  // memory lives on the GPU
    pool_props.location.id = device;                       // GPU to allocate on
    cudaMemPool_t pool{};
    CHECK_CUDA(cudaMemPoolCreate(&pool, &pool_props));

    // configure memory pool to never release back to os
    uint64_t threshold = UINT64_MAX;
    CHECK_CUDA(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    return pool;
  }

  uint8_t device_;
  cudaMemPool_t memory_pool_{};
};

}  // namespace fast_deconv::core
