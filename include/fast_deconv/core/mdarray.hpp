#pragma once

#include <cuda/std/mdspan>
#include <cuda_runtime.h>

#include <fast_deconv/core/memory_policy.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::core {

template <class T, class Extents, class LayoutPolicy, class MemoryPolicy>
class MDArray {
  using element_type = T;
  using extents_type = Extents;
  using layout_type  = LayoutPolicy;
  using span_type    = cuda::std::mdspan<T, Extents, LayoutPolicy>;

 public:
  explicit MDArray(extents_type exts) : exts_(exts)
  {
    std::size_t total = 1;
    for (int i = 0; i < static_cast<int>(exts_.rank()); ++i)
      total *= exts_.extent(i);
    MemoryPolicy::allocate(ptr_, total);
    view_ = span_type(ptr_, exts_);
  }
  ~MDArray() { MemoryPolicy::free(this->ptr_); };

  span_type view() noexcept { return this->view_; }

 private:
  T* ptr_ = nullptr;
  extents_type exts_{};
  span_type view_{};
};

template <typename T, typename MemoryPolicy, typename... Exts>
auto make_mdarray(Exts... exts)
{
  static_assert((std::is_integral_v<Exts> && ...), "extents must be integers");
  constexpr std::size_t rank = sizeof...(Exts);
  using extents_t            = cuda::std::dextents<std::size_t, rank>;
  using layout_t             = cuda::std::layout_right;
  using mem_t                = MemoryPolicy;
  return MDArray<T, extents_t, layout_t, mem_t>(extents_t(static_cast<std::size_t>(exts)...));
}

template <typename T, typename... Exts>
auto make_device_mdarray(Exts... exts)
{
  return make_mdarray<T, DeviceMemory<T>>(exts...);
}

template <typename T, typename... Exts>
auto make_managed_mdarray(Exts... exts)
{
  return make_mdarray<T, ManagedMemory<T>>(exts...);
}

}  // namespace fast_deconv::core
