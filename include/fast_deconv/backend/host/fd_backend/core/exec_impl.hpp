#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fast_deconv/core/profiler.hpp>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace fast_deconv::core {

/**
 * @brief Host counterpart of the cuda memory pool: whole blocks only, no splitting.
 *
 * A freed block goes on the free list for its exact size and is handed back on the next request for that
 * size, returned to the OS only by trim(). The scratch sizes repeat identically every iteration, so exact-size
 * matching hits every time and the multi-GB mmap/munmap round trip disappears with it.
 */
class block_pool {
 public:
  /// Matches the alignment device allocations give the vectorized paths.
  static constexpr std::align_val_t alignment{64};

  block_pool() = default;

  block_pool(const block_pool&) = delete;
  block_pool& operator=(const block_pool&) = delete;
  block_pool(block_pool&&) = delete;
  block_pool& operator=(block_pool&&) = delete;

  ~block_pool() { trim(); }

  /// Returns every free-list block to the OS; live blocks are untouched.
  void trim() noexcept
  {
    const std::lock_guard lock(mutex_);
    for (const auto& [num_bytes, blocks] : free_lists_) {
      for (void* ptr : blocks) ::operator delete(ptr, alignment);
    }
    free_lists_.clear();
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
    peak_used_bytes_ = std::max(peak_used_bytes_, used_bytes_);
    // Under the lock: Tracy needs the reported order to match the real one.
    FD_PROFILE_ALLOC(ptr, num_bytes);
    return ptr;
  }

  /**
   * @brief Returns @p ptr to the free list for its size.
   *
   * The free list is the block's only owner once it is back, so a push that throws has to release it
   * rather than drop it on the floor.
   *
   * @param[in] ptr  Block from alloc(); a pointer the pool does not track is deleted outright.
   */
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
    FD_PROFILE_FREE(ptr);

    try {
      free_lists_[num_bytes].push_back(ptr);
    } catch (...) {
      ::operator delete(ptr, alignment);
    }
  }

  /**
   * @brief Bytes currently allocated by user code (live working set).
   *
   * Not the bytes the pool holds — the same thing the cuda backend reports.
   */
  std::uint64_t used_bytes() const
  {
    const std::lock_guard lock(mutex_);
    return used_bytes_;
  }

  /// Bytes the pool holds: live working set plus the blocks waiting in the free lists.
  std::uint64_t reserved_bytes() const
  {
    const std::lock_guard lock(mutex_);
    std::uint64_t idle = 0;
    for (const auto& [num_bytes, blocks] : free_lists_) idle += num_bytes * blocks.size();
    return used_bytes_ + idle;
  }

  /// Highest used_bytes() since the pool was created.
  std::uint64_t peak_used_bytes() const
  {
    const std::lock_guard lock(mutex_);
    return peak_used_bytes_;
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::uint64_t, std::vector<void*>> free_lists_;
  std::unordered_map<void*, std::uint64_t> live_;
  std::uint64_t used_bytes_ = 0;
  std::uint64_t peak_used_bytes_ = 0;
};

/**
 * @brief Owns the host memory pool.
 *
 * No device to own, but it owns the pool and hands it to every exec_ctx created from it, the same shape as
 * the cuda backend.
 */
class exec_resources_impl {
 public:
  explicit exec_resources_impl(std::uint8_t /*device*/) {}

  exec_resources_impl(const exec_resources_impl&) = delete;
  exec_resources_impl& operator=(const exec_resources_impl&) = delete;
  exec_resources_impl(exec_resources_impl&&) = delete;
  exec_resources_impl& operator=(exec_resources_impl&&) = delete;

  std::uint64_t pool_used_bytes() const { return memory_pool.used_bytes(); }
  std::uint64_t pool_reserved_bytes() const { return memory_pool.reserved_bytes(); }
  std::uint64_t pool_peak_used_bytes() const { return memory_pool.peak_used_bytes(); }

  /// Returns every idle block to the OS.
  void trim() const { memory_pool.trim(); }

  /**
   * @brief Pool every exec_ctx allocates from.
   *
   * mutable so an exec_ctx built from a const resources still allocates through it, the way the cuda exec_ctx_impl
   * copies its pool handle out of one.
   */
  mutable block_pool memory_pool;
};

/**
 * @brief The host exec_ctx.
 *
 * Work runs inline, so wait() has nothing to wait for and allocation is a pool hit.
 */
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

  void copy_bytes(void* dst, const void* src, std::uint64_t num_bytes) const { std::memcpy(dst, src, num_bytes); }

  void zero_bytes(void* dst, std::uint64_t num_bytes) const { std::memset(dst, 0, num_bytes); }

  void wait() const {}

  block_pool& memory_pool;
};

}  // namespace fast_deconv::core
