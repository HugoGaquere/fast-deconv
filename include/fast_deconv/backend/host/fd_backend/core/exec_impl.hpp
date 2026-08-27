#pragma once

#include <cstdint>
#include <cstring>
#include <new>

namespace fast_deconv::core {

/// No device and no pool to own; kept so both backends compose the same way.
class exec_resources_impl {
 public:
  explicit exec_resources_impl(std::uint8_t /*device*/) {}

  exec_resources_impl(const exec_resources_impl&) = delete;
  exec_resources_impl& operator=(const exec_resources_impl&) = delete;
  exec_resources_impl(exec_resources_impl&&) = delete;
  exec_resources_impl& operator=(exec_resources_impl&&) = delete;

  std::uint64_t pool_used_bytes() const { return 0; }
};

/// One execution lane on the host. Work runs inline, so wait() has nothing to
/// wait for and allocation is a plain aligned new.
class exec_ctx_impl {
 public:
  /// Matches the alignment device allocations give the vectorized paths.
  static constexpr std::align_val_t alignment{64};

  explicit exec_ctx_impl(const exec_resources_impl& /*res*/) {}

  exec_ctx_impl(const exec_ctx_impl&) = delete;
  exec_ctx_impl& operator=(const exec_ctx_impl&) = delete;
  exec_ctx_impl(exec_ctx_impl&&) = delete;
  exec_ctx_impl& operator=(exec_ctx_impl&&) = delete;

  void* alloc_bytes(std::uint64_t num_bytes) const { return ::operator new(num_bytes, alignment); }

  void free_bytes(void* ptr) const noexcept
  {
    if (ptr != nullptr) ::operator delete(ptr, alignment);
  }

  void copy_from_host_bytes(void* dst, const void* src, std::uint64_t num_bytes) const
  {
    std::memcpy(dst, src, num_bytes);
  }

  void copy_to_host_bytes(void* dst, const void* src, std::uint64_t num_bytes) const
  {
    std::memcpy(dst, src, num_bytes);
  }

  void wait() const {}
};

}  // namespace fast_deconv::core
