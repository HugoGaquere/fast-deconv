#pragma once

#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <type_traits>
#include <vector>

namespace fast_deconv::test {

// Owning allocation in backend memory on lane @p sr, freed (with a wait) on
// destruction — an early ASSERT return cannot leak pool memory. Every transfer
// goes through core::exec_ctx's backend-neutral primitives, so the same test
// source compiles against the cuda and host backends.
template <typename T>
class device_buffer {
 public:
  device_buffer(const core::exec_ctx& sr, std::size_t n) : sr_(sr), n_(n), ptr_(sr.alloc_async<T>(n)) {}

  // Upload constructor. std::vector<bool> is bit-packed, so it goes through a
  // uint8_t staging buffer; either way the copy completes before returning.
  device_buffer(const core::exec_ctx& sr, const std::vector<T>& host) : device_buffer(sr, host.size())
  {
    if constexpr (std::is_same_v<T, bool>) {
      std::vector<uint8_t> bytes(host.size());
      for (std::size_t i = 0; i < host.size(); ++i) bytes.at(i) = host.at(i) ? 1 : 0;
      sr_.copy_from_host_bytes(ptr_, bytes.data(), n_ * sizeof(bool));
    } else {
      sr_.copy_from_host_bytes(ptr_, host.data(), n_ * sizeof(T));
    }
    sr_.wait();
  }

  ~device_buffer()
  {
    sr_.free_async(ptr_);
    sr_.wait();
  }

  device_buffer(const device_buffer&) = delete;
  device_buffer& operator=(const device_buffer&) = delete;
  device_buffer(device_buffer&&) = delete;
  device_buffer& operator=(device_buffer&&) = delete;

  T* get() const { return ptr_; }
  std::size_t size() const { return n_; }

  // Blocking read back to the host. Returns std::vector<uint8_t> for bool
  // buffers to sidestep std::vector<bool> bit-packing.
  auto to_host() const
  {
    using element = std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>;
    std::vector<element> host(n_);
    sr_.copy_to_host_bytes(host.data(), ptr_, n_ * sizeof(T));
    sr_.wait();
    return host;
  }

  // Blocking refresh of an existing buffer (non-bool only).
  void from_host(const std::vector<T>& host)
  {
    static_assert(!std::is_same_v<T, bool>, "use the upload constructor for bool buffers");
    sr_.copy_from_host_bytes(ptr_, host.data(), host.size() * sizeof(T));
    sr_.wait();
  }

  // Sets every byte to @p value — the neutral stand-in for a device memset,
  // used to poison an output buffer before the code under test fills it.
  void fill_bytes(int value)
  {
    const std::vector<uint8_t> bytes(n_ * sizeof(T), static_cast<uint8_t>(value));
    sr_.copy_from_host_bytes(ptr_, bytes.data(), bytes.size());
    sr_.wait();
  }

 private:
  const core::exec_ctx& sr_;
  std::size_t n_;
  T* ptr_;
};

// Allocates required_work_size() bytes, binds them to @p conv, and frees them on
// scope exit. Every convolve_ctx in a test goes through this, so a plan can
// never execute with an unbound work area (a recurring bug in tests: see
// commits b0bfdca and 65addc8).
class scoped_work_area {
 public:
  scoped_work_area(const core::exec_ctx& sr, linalg::convolve_ctx& conv) : sr_(sr)
  {
    if (conv.required_work_size() > 0) ptr_ = sr.alloc_async(conv.required_work_size());
    conv.bind_work_area(ptr_);
  }

  ~scoped_work_area()
  {
    sr_.free_async(ptr_);
    sr_.wait();
  }

  scoped_work_area(const scoped_work_area&) = delete;
  scoped_work_area& operator=(const scoped_work_area&) = delete;

 private:
  const core::exec_ctx& sr_;
  void* ptr_ = nullptr;
};

}  // namespace fast_deconv::test
