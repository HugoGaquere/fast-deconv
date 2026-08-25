#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <emu/cuda/device/mdcontainer.hpp>
#include <emu/cuda/memory.hpp>
#include <emu/cuda/stream.hpp>
#include <memory>
#include <stdexcept>

#include "cublas_v2.h"
#include "fast_deconv/core/span_types.hpp"
#include "fast_deconv/util/cublas_macros.hpp"
#include "fast_deconv/util/cuda_macros.hpp"

namespace fast_deconv::core {

struct stream_deleter {
  cudaStream_t stream{};
  void operator()(void* ptr) const noexcept
  {
    if (ptr != nullptr) CHECK_CUDA(cudaFreeAsync(ptr, stream));
  }
};

template <typename T>
using device_ptr = std::unique_ptr<T[], stream_deleter>;

class stream_resources {
 public:
  static constexpr uint default_flag{cudaStreamDefault};

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
  [[nodiscard]] T* alloc_async(uint64_t n) const
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

  template <typename T, typename... Exts>
    requires((std::convertible_to<Exts, std::int32_t> && ...) && sizeof...(Exts) > 0)
  [[nodiscard]] mdcontainer<T, sizeof...(Exts)> alloc_mdcontainer_async(Exts... exts) const
  {
    const uint64_t n = (uint64_t{1} * ... * static_cast<uint64_t>(exts));
    T* ptr = alloc_async<T>(n);
    return mdcontainer<T, sizeof...(Exts)>(ptr, device_ptr<T>(ptr, stream_deleter{cuda_stream}), emu::exts_flag,
                                           exts...);
  }

  template <typename T, typename Extents>
    requires requires { Extents::rank(); }
  [[nodiscard]] mdcontainer<T, Extents::rank()> alloc_mdcontainer_async(const Extents& exts) const
  {
    uint64_t n = 1;
    for (std::size_t i = 0; i < Extents::rank(); ++i) n *= static_cast<uint64_t>(exts.extent(i));
    T* ptr = alloc_async<T>(n);
    return mdcontainer<T, Extents::rank()>(ptr, device_ptr<T>(ptr, stream_deleter{cuda_stream}),
                                           dims<Extents::rank()>(exts));
  }

  template <typename T>
  [[nodiscard]] device_ptr<T> alloc_ptr_async(uint64_t n) const
  {
    return {alloc_async<T>(n), stream_deleter{cuda_stream}};
  }

  template <typename HostSpan>
  [[nodiscard]] auto copy_h2d_async(const HostSpan& src) const
  {
    using T = typename HostSpan::element_type;
    auto dst = alloc_mdcontainer_async<T>(src.extents());
    CHECK_CUDA(cudaMemcpyAsync(dst.data_handle(), src.data_handle(), src.size() * sizeof(T), cudaMemcpyHostToDevice,
                               cuda_stream));
    return dst;
  }

  template <typename T>
  void free_async(T* ptr) const
  {
    if (ptr == nullptr) return;
    CHECK_CUDA(cudaFreeAsync(ptr, cuda_stream));
  }

  void sync() const { CHECK_CUDA(cudaStreamSynchronize(cuda_stream)); }

  cudaStream_t cuda_stream{};
  cublasHandle_t cublas_handle{};
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

  [[nodiscard]] stream_resources make_stream(uint flag = stream_resources::default_flag) const
  {
    return {memory_pool_, flag, device_};
  }

  template <typename T = void>
  [[nodiscard]] T* alloc_async(uint64_t n, const stream_resources& stream_r) const
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
  [[nodiscard]] uint64_t pool_used_bytes() const
  {
    cuuint64_t v = 0;
    CHECK_CUDA(cudaMemPoolGetAttribute(memory_pool_, cudaMemPoolAttrUsedMemCurrent, &v));
    return static_cast<uint64_t>(v);
  }

 private:
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
