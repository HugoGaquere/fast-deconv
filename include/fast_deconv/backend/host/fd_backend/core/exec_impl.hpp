#pragma once

#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace fast_deconv::core {

/// Host counterpart of the cuda memory pool: whole blocks only, no splitting.
/// A freed block goes on the free list for its exact size and is handed back on
/// the next request for that size, never returned to the OS. The scratch sizes
/// repeat identically every iteration, so exact-size matching hits every time
/// and the multi-GB mmap/munmap round trip disappears with it.
class block_pool {
 public:
  /// Matches the alignment device allocations give the vectorized paths.
  static constexpr std::align_val_t alignment{64};

  block_pool() = default;

  block_pool(const block_pool&) = delete;
  block_pool& operator=(const block_pool&) = delete;
  block_pool(block_pool&&) = delete;
  block_pool& operator=(block_pool&&) = delete;

  ~block_pool()
  {
    for (const auto& [num_bytes, blocks] : free_lists_) {
      for (void* ptr : blocks) ::operator delete(ptr, alignment);
    }
  }

  void* alloc(std::uint64_t num_bytes)
  {
    const std::lock_guard lock(mutex_);

    auto& blocks = free_lists_[num_bytes];
    void* ptr = nullptr;
    if (blocks.empty()) {
      ptr = ::operator new(num_bytes, alignment);
    } else {
      ptr = blocks.back();
      blocks.pop_back();
    }

    try {
      live_[ptr] = num_bytes;
    } catch (...) {
      // Neither the caller nor the free list owns this block yet.
      ::operator delete(ptr, alignment);
      throw;
    }
    used_bytes_ += num_bytes;
    return ptr;
  }

  /// The free list is the block's only owner once it is back, so a push that
  /// throws has to release it rather than drop it on the floor.
  void recycle(void* ptr) noexcept
  {
    const std::lock_guard lock(mutex_);

    const auto it = live_.find(ptr);
    if (it == live_.end()) {
      ::operator delete(ptr, alignment);
      return;
    }
    const std::uint64_t num_bytes = it->second;
    live_.erase(it);
    used_bytes_ -= num_bytes;

    try {
      free_lists_[num_bytes].push_back(ptr);
    } catch (...) {
      ::operator delete(ptr, alignment);
    }
  }

  /// Bytes currently allocated by user code (live working set), not the bytes
  /// the pool holds — the same thing the cuda backend reports.
  std::uint64_t used_bytes() const
  {
    const std::lock_guard lock(mutex_);
    return used_bytes_;
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::uint64_t, std::vector<void*>> free_lists_;
  std::unordered_map<void*, std::uint64_t> live_;
  std::uint64_t used_bytes_ = 0;
};

/// No device to own, but it owns the pool and hands it to every lane created
/// from it, the same shape as the cuda backend.
class exec_resources_impl {
 public:
  explicit exec_resources_impl(std::uint8_t /*device*/) {}

  exec_resources_impl(const exec_resources_impl&) = delete;
  exec_resources_impl& operator=(const exec_resources_impl&) = delete;
  exec_resources_impl(exec_resources_impl&&) = delete;
  exec_resources_impl& operator=(exec_resources_impl&&) = delete;

  std::uint64_t pool_used_bytes() const { return memory_pool.used_bytes(); }

  /// mutable so a lane built from a const resources still allocates through it,
  /// the way the cuda lane copies its pool handle out of one.
  mutable block_pool memory_pool;
};

/// One execution lane on the host. Work runs inline, so wait() has nothing to
/// wait for and allocation is a pool hit.
class exec_ctx_impl {
 public:
  static constexpr std::align_val_t alignment = block_pool::alignment;

  /// Backend memory is host memory, so stage() can borrow instead of copying.
  static constexpr bool host_resident = true;

  explicit exec_ctx_impl(const exec_resources_impl& res) : memory_pool(res.memory_pool) {}

  exec_ctx_impl(const exec_ctx_impl&) = delete;
  exec_ctx_impl& operator=(const exec_ctx_impl&) = delete;
  exec_ctx_impl(exec_ctx_impl&&) = delete;
  exec_ctx_impl& operator=(exec_ctx_impl&&) = delete;

  void* alloc_bytes(std::uint64_t num_bytes) const { return memory_pool.alloc(num_bytes); }

  void free_bytes(void* ptr) const noexcept
  {
    if (ptr != nullptr) memory_pool.recycle(ptr);
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

  block_pool& memory_pool;
};

}  // namespace fast_deconv::core
