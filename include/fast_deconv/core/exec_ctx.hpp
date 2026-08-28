#pragma once
#include <concepts>
#include <cstdint>
#include <fast_deconv/core/memory_types.hpp>
#include <memory>
#include <stdexcept>

// Resolved by the include path: CMake puts backend/${FAST_DECONV_BACKEND}
#include <fd_backend/core/exec_impl.hpp>

namespace fast_deconv::core {

struct ctx_deleter {
  const exec_ctx_impl* ctx{};
  void operator()(void* ptr) const noexcept
  {
    if (ctx != nullptr) ctx->free_bytes(ptr);
  }
};

template <typename T>
using owned_ptr = std::unique_ptr<T[], ctx_deleter>;

/// One execution lane: where work runs and where its memory comes from.
/// The backend supplies the handles and the five primitives below it; the
/// typed layer here is written once and shared by every backend.
class exec_ctx : public exec_ctx_impl {
 public:
  using impl = exec_ctx_impl;

  template <typename T = void>
  T* alloc_async(std::uint64_t n) const
  {
    if (n == 0) throw std::invalid_argument("alloc_async: n must be > 0");

    std::uint64_t num_bytes;
    if constexpr (std::is_void_v<T>)
      num_bytes = n;
    else
      num_bytes = n * sizeof(T);

    return static_cast<T*>(this->alloc_bytes(num_bytes));
  }

  template <typename T, typename... Exts>
    requires((std::convertible_to<Exts, std::int32_t> && ...) && sizeof...(Exts) > 0)
  mdcontainer<T, sizeof...(Exts)> alloc_mdcontainer_async(Exts... exts) const
  {
    const std::uint64_t n = (std::uint64_t{1} * ... * static_cast<std::uint64_t>(exts));
    T* ptr = alloc_async<T>(n);
    return mdcontainer<T, sizeof...(Exts)>(ptr, owned_ptr<T>(ptr, ctx_deleter{this}), emu::exts_flag, exts...);
  }

  template <typename T, typename Extents>
    requires requires { Extents::rank(); }
  mdcontainer<T, Extents::rank()> alloc_mdcontainer_async(const Extents& exts) const
  {
    std::uint64_t n = 1;
    for (std::size_t i = 0; i < Extents::rank(); ++i) n *= static_cast<std::uint64_t>(exts.extent(i));
    T* ptr = alloc_async<T>(n);
    return mdcontainer<T, Extents::rank()>(ptr, owned_ptr<T>(ptr, ctx_deleter{this}), dims<Extents::rank()>(exts));
  }

  template <typename T>
  owned_ptr<T> alloc_ptr_async(std::uint64_t n) const
  {
    return {alloc_async<T>(n), ctx_deleter{this}};
  }

  /// Stages caller-owned host memory into backend memory. A real copy on every
  /// backend, so ownership works out the same whichever one is built.
  template <typename HostSpan>
  auto upload(const HostSpan& src) const
  {
    using T = typename HostSpan::element_type;
    auto dst = alloc_mdcontainer_async<T>(src.extents());
    this->copy_from_host_bytes(dst.data_handle(), src.data_handle(), src.size() * sizeof(T));
    return dst;
  }

  /// Copies backend memory back into caller-owned host memory. Async like
  /// upload(), so wait() before reading @p host_dst.
  template <typename Span>
  void download(const Span& src, typename Span::element_type* host_dst) const
  {
    using T = typename Span::element_type;
    this->copy_to_host_bytes(host_dst, src.data_handle(), src.size() * sizeof(T));
  }

  template <typename T>
  void free_async(T* ptr) const
  {
    this->free_bytes(ptr);
  }

 private:
  friend class exec_resources;
  explicit exec_ctx(const exec_resources_impl& res) : impl(res) {}
};

/// Owns whatever the backend executes on and allocates from
class exec_resources : public exec_resources_impl {
 public:
  using impl = exec_resources_impl;
  using impl::impl;

  exec_ctx make_ctx() const { return exec_ctx(*this); }
};

}  // namespace fast_deconv::core
