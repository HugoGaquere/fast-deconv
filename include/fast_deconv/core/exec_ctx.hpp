#pragma once
#include <cassert>
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

/**
 * @brief One execution context: where work runs and where its memory comes from.
 *
 * The backend supplies the handles and the primitives below it; the typed layer here is written once and
 * shared by every backend.
 */
class exec_ctx : public exec_ctx_impl {
 public:
  using impl = exec_ctx_impl;

  /**
   * @brief Uninitialized backend memory for @p n elements of T (bytes for void), from the pool.
   *
   * Stream-ordered on CUDA: usable by work enqueued on this context after the call. Release with free(); the
   * typed helpers below do it on destruction.
   *
   * @throws std::invalid_argument If @p n <= 0.
   */
  template <typename T = void>
  T* alloc(std::int64_t n) const
  {
    if (n <= 0) throw std::invalid_argument("alloc: n must be > 0");

    std::size_t num_bytes = narrow<std::size_t>(n);
    if constexpr (!std::is_void_v<T>) num_bytes *= sizeof(T);

    return static_cast<T*>(this->alloc_bytes(num_bytes));
  }

  /// alloc() of a row-major container with extents @p exts, freed through this context when its last copy
  /// goes.
  template <typename T, std::integral... Exts>
    requires(sizeof...(Exts) > 0)
  mdcontainer<T, sizeof...(Exts)> alloc_mdcontainer(Exts... exts) const
  {
    return alloc_mdcontainer<T>(dims<sizeof...(Exts)>(narrow<std::int64_t>(exts)...));
  }

  /// alloc_mdcontainer() taking the extents as one object (e.g. another view's extents()).
  template <typename T, typename Extents>
    requires requires { Extents::rank(); }
  mdcontainer<T, Extents::rank()> alloc_mdcontainer(const Extents& exts) const
  {
    const dims<Extents::rank()> d(exts);
    std::int64_t n = 1;
    for (std::size_t i = 0; i < Extents::rank(); ++i) n *= d.extent(i);
    T* ptr = alloc<T>(n);
    return mdcontainer<T, Extents::rank()>(ptr, owned_ptr<T>(ptr, ctx_deleter{this}), d);
  }

  /// alloc() of @p n elements, freed through this context when the pointer goes.
  template <typename T>
  owned_ptr<T> alloc_ptr(std::int64_t n) const
  {
    return {alloc<T>(n), ctx_deleter{this}};
  }

  /**
   * @brief Copies @p src into @p dst; either side may be host or backend memory.
   *
   * Async, so wait() before reading @p dst.
   *
   * @param[out] dst  Destination view, same size as @p src.
   * @param[in]  src  Source view.
   * @throws std::invalid_argument If the sizes differ.
   */
  template <typename Dst, typename Src>
  void copy(const Dst& dst, const Src& src) const
  {
    if (dst.size() != src.size()) throw std::invalid_argument("copy: dst and src sizes differ");
    copy(dst.data_handle(), src.data_handle(), narrow<std::int64_t>(src.size()));
  }

  /// Raw-pointer copy() of @p n elements.
  template <typename T>
  void copy(T* dst, const T* src, std::int64_t n) const
  {
    this->copy_bytes(dst, src, narrow<std::size_t>(n) * sizeof(T));
  }

  /**
   * @brief Backend-resident view of caller-owned host memory.
   *
   * A copy on a device backend, a borrow when backend memory already is host memory.
   *
   * @param[in] src  Host view. Must outlive the returned view.
   * @return A view with @p src's element type and extents, in backend memory.
   */
  template <typename HostSpan>
  auto stage(const HostSpan& src) const
  {
    using T = typename HostSpan::element_type;
    constexpr std::size_t rank = HostSpan::rank();

    if constexpr (impl::host_resident) {
      return mdcontainer<T, rank>(src.data_handle(), emu::capsule{}, src.extents());
    } else {
      // The staging buffer is written, so it is allocated mutable; the view handed
      // back keeps the caller's element type, so a const input stays unwritable.
      auto owner = alloc_mdcontainer<std::remove_const_t<T>>(src.extents());
      copy(owner, src);
      return mdcontainer<T, rank>(owner.data_handle(), std::move(owner).capsule(), src.extents());
    }
  }

  /**
   * @brief Undoes stage() for an in/out buffer.
   *
   * A copy() where stage() copied, nothing where it borrowed and the kernels already wrote @p host_dst.
   *
   * @param[in]  src       View returned by stage().
   * @param[out] host_dst  The host buffer that was staged.
   */
  template <typename Span>
  void unstage(const Span& src, typename Span::element_type* host_dst) const
  {
    if constexpr (impl::host_resident)
      assert(src.data_handle() == host_dst && "unstage() target is not the buffer stage() borrowed");
    else
      copy(host_dst, src.data_handle(), narrow<std::int64_t>(src.size()));
  }

  /// Returns memory from alloc() to the pool; stream-ordered on CUDA, after the work already enqueued here.
  template <typename T>
  void free(T* ptr) const
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

  /// A new execution context (on CUDA, its own stream) allocating from this pool. Must not outlive this.
  exec_ctx make_ctx() const { return exec_ctx(*this); }
};

}  // namespace fast_deconv::core
