#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "cublas_v2.h"
#include "fast_deconv/util/cublas_macros.hpp"
#include "fast_deconv/util/cuda_macros.hpp"

namespace fast_deconv::core {

/**
 * @brief Owns the device and its memory pool.
 *
 * One per device; hands its pool to every exec_ctx created from it.
 */
class exec_resources_impl {
 public:
  explicit exec_resources_impl(std::uint8_t device) : device(device), memory_pool(make_memory_pool(device)) {}

  exec_resources_impl(const exec_resources_impl&) = delete;
  exec_resources_impl& operator=(const exec_resources_impl&) = delete;
  exec_resources_impl(exec_resources_impl&&) = delete;
  exec_resources_impl& operator=(exec_resources_impl&&) = delete;

  ~exec_resources_impl() { CHECK_CUDA(cudaMemPoolDestroy(memory_pool)); }

  /// Bytes currently allocated by user code (live working set).
  std::uint64_t pool_used_bytes() const
  {
    cuuint64_t v = 0;
    CHECK_CUDA(cudaMemPoolGetAttribute(memory_pool, cudaMemPoolAttrUsedMemCurrent, &v));
    return static_cast<std::uint64_t>(v);
  }

  /// Bytes the pool holds from the driver: live working set plus idle, reusable blocks.
  std::uint64_t pool_reserved_bytes() const
  {
    cuuint64_t v = 0;
    CHECK_CUDA(cudaMemPoolGetAttribute(memory_pool, cudaMemPoolAttrReservedMemCurrent, &v));
    return static_cast<std::uint64_t>(v);
  }

  std::uint8_t device{};
  cudaMemPool_t memory_pool{};

 private:
  static cudaMemPool_t make_memory_pool(std::uint8_t device)
  {
    cudaMemPoolProps pool_props = {};
    pool_props.allocType = cudaMemAllocationTypePinned;    // non-migratable device memory
    pool_props.handleTypes = cudaMemHandleTypeNone;        // no inter-process sharing
    pool_props.location.type = cudaMemLocationTypeDevice;  // memory lives on the GPU
    pool_props.location.id = device;                       // GPU to allocate on
    cudaMemPool_t pool{};
    CHECK_CUDA(cudaMemPoolCreate(&pool, &pool_props));

    // Never release freed blocks back to the driver: they stay in the pool for reuse.
    std::uint64_t threshold = UINT64_MAX;
    CHECK_CUDA(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    return pool;
  }
};

/**
 * @brief The cuda exec_ctx: a stream, its cuBLAS handle, and stream-ordered allocation from the owning pool.
 *
 * Members are public because the kernel launches in the .cu files need the stream.
 */
class exec_ctx_impl {
 public:
  /// Backend memory lives on the device, so stage() must really copy.
  static constexpr bool host_resident = false;

  explicit exec_ctx_impl(const exec_resources_impl& res) : device(res.device), memory_pool(res.memory_pool)
  {
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamDefault));
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, cuda_stream));
  }

  exec_ctx_impl(const exec_ctx_impl&) = delete;
  exec_ctx_impl& operator=(const exec_ctx_impl&) = delete;
  exec_ctx_impl(exec_ctx_impl&&) = delete;
  exec_ctx_impl& operator=(exec_ctx_impl&&) = delete;

  ~exec_ctx_impl()
  {
    CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaStreamDestroy(cuda_stream));
  }

  void* alloc_bytes(std::uint64_t num_bytes) const
  {
    void* ptr = nullptr;
    CHECK_CUDA(cudaMallocFromPoolAsync(&ptr, num_bytes, memory_pool, cuda_stream));
    return ptr;
  }

  void free_bytes(void* ptr) const noexcept
  {
    if (ptr != nullptr) CHECK_CUDA(cudaFreeAsync(ptr, cuda_stream));
  }

  // cudaMemcpyDefault: unified addressing infers the direction from the pointers.
  void copy_bytes(void* dst, const void* src, std::uint64_t num_bytes) const
  {
    CHECK_CUDA(cudaMemcpyAsync(dst, src, num_bytes, cudaMemcpyDefault, cuda_stream));
  }

  void zero_bytes(void* dst, std::uint64_t num_bytes) const
  {
    CHECK_CUDA(cudaMemsetAsync(dst, 0, num_bytes, cuda_stream));
  }

  void wait() const { CHECK_CUDA(cudaStreamSynchronize(cuda_stream)); }

  cudaStream_t cuda_stream{};
  cublasHandle_t cublas_handle{};
  std::uint8_t device{};
  cudaMemPool_t memory_pool{};
};

}  // namespace fast_deconv::core
