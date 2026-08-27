#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <type_traits>
#include <vector>

namespace fast_deconv::test {

// Owning device allocation from @p res's pool on stream @p sr, freed (with a
// stream sync) on destruction — an early ASSERT return cannot leak pool memory.
template <typename T>
class device_buffer {
 public:
  // `res` is kept in the signature (110 call sites) but unused: allocation moved
  // from the pool owner onto the lane when exec_ctx landed.
  device_buffer(const core::resources& /*res*/, const core::stream_resources& sr, std::size_t n)
      : sr_(sr), n_(n), ptr_(sr.alloc_async<T>(n))
  {
  }

  // Upload constructor. std::vector<bool> is bit-packed, so it goes through a
  // uint8_t staging buffer; either way the copy completes before returning.
  device_buffer(const core::resources& res, const core::stream_resources& sr, const std::vector<T>& host)
      : device_buffer(res, sr, host.size())
  {
    if constexpr (std::is_same_v<T, bool>) {
      std::vector<uint8_t> bytes(host.size());
      for (std::size_t i = 0; i < host.size(); ++i) bytes.at(i) = host.at(i) ? 1 : 0;
      CHECK_CUDA(cudaMemcpyAsync(ptr_, bytes.data(), n_ * sizeof(bool), cudaMemcpyHostToDevice, sr_.cuda_stream));
    } else {
      CHECK_CUDA(cudaMemcpyAsync(ptr_, host.data(), n_ * sizeof(T), cudaMemcpyHostToDevice, sr_.cuda_stream));
    }
    sr_.sync();
  }

  ~device_buffer()
  {
    sr_.free_async(ptr_);
    sr_.sync();
  }

  device_buffer(const device_buffer&) = delete;
  device_buffer& operator=(const device_buffer&) = delete;
  device_buffer(device_buffer&&) = delete;
  device_buffer& operator=(device_buffer&&) = delete;

  T* get() const { return ptr_; }
  std::size_t size() const { return n_; }

  // Blocking device-to-host copy. Returns std::vector<uint8_t> for bool
  // buffers to sidestep std::vector<bool> bit-packing.
  auto to_host() const
  {
    if constexpr (std::is_same_v<T, bool>) {
      std::vector<uint8_t> host(n_);
      CHECK_CUDA(cudaMemcpyAsync(host.data(), ptr_, n_ * sizeof(bool), cudaMemcpyDeviceToHost, sr_.cuda_stream));
      sr_.sync();
      return host;
    } else {
      std::vector<T> host(n_);
      CHECK_CUDA(cudaMemcpyAsync(host.data(), ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost, sr_.cuda_stream));
      sr_.sync();
      return host;
    }
  }

  // Blocking host-to-device refresh of an existing buffer (non-bool only).
  void from_host(const std::vector<T>& host)
  {
    static_assert(!std::is_same_v<T, bool>, "use the upload constructor for bool buffers");
    CHECK_CUDA(cudaMemcpyAsync(ptr_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, sr_.cuda_stream));
    sr_.sync();
  }

 private:
  const core::stream_resources& sr_;
  std::size_t n_;
  T* ptr_;
};

// Blocking download of a raw device pointer written by library code.
template <typename T>
std::vector<T> download(const core::stream_resources& sr, const T* d_ptr, std::size_t n)
{
  std::vector<T> host(n);
  CHECK_CUDA(cudaMemcpyAsync(host.data(), d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost, sr.cuda_stream));
  sr.sync();
  return host;
}

inline std::vector<uint8_t> download_bool(const core::stream_resources& sr, const bool* d_ptr, std::size_t n)
{
  std::vector<uint8_t> host(n);
  CHECK_CUDA(cudaMemcpyAsync(host.data(), d_ptr, n * sizeof(bool), cudaMemcpyDeviceToHost, sr.cuda_stream));
  sr.sync();
  return host;
}

// Allocates required_work_size() bytes, binds them to @p ctx, and frees them on
// scope exit. Every convolve_ctx in a test goes through this, so a plan can
// never execute with an unbound cuFFT work area (a recurring bug in tests: see
// commits b0bfdca and 65addc8).
class scoped_work_area {
 public:
  scoped_work_area(const core::resources& /*res*/, const core::stream_resources& sr, linalg::convolve_ctx& conv)
      : sr_(sr)
  {
    if (conv.required_work_size() > 0) ptr_ = sr.alloc_async(conv.required_work_size());
    conv.bind_work_area(ptr_);
  }

  ~scoped_work_area()
  {
    sr_.free_async(ptr_);
    sr_.sync();
  }

  scoped_work_area(const scoped_work_area&) = delete;
  scoped_work_area& operator=(const scoped_work_area&) = delete;

 private:
  const core::stream_resources& sr_;
  void* ptr_ = nullptr;
};

}  // namespace fast_deconv::test
