#pragma once
#include <emu/cuda/device/mdspan.hpp>
#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/core/kernel_traits.hpp>

namespace fast_deconv::core {

template <core::cpts::mdspan Mdspan>
constexpr bool inner_unit_stride(const Mdspan& m)
{
  // fmt::println("strides {} {}", m.mapping().stride(0), m.mapping().stride(1));
  if constexpr (cpts::is_layout_right<Mdspan>) {
    return true;
  } else if constexpr (cpts::is_layout_left<Mdspan>) {
    return false;
  } else {
    constexpr int R = Mdspan::rank();
    return m.mapping().stride(R - 1) == 1;
  }
}

template <core::cpts::mdspan Mdspan>
bool rows_aligned_for_vec(const Mdspan& m, std::size_t vec_bytes)
{
  if constexpr (cpts::is_layout_right<Mdspan>) {
    return true;
  } else if constexpr (!cpts::is_layout_stride<Mdspan>) {
    return false;
  } else {
    constexpr int R = Mdspan::rank();
    constexpr std::size_t elem_size = sizeof(typename Mdspan::element_type);
    for (int d = 0; d < R - 1; ++d) {
      const std::size_t stride_bytes = static_cast<std::size_t>(m.mapping().stride(d)) * elem_size;
      if ((stride_bytes & (vec_bytes - 1)) != 0) return false;
    }
    return true;
  }
}

template <cpts::mdspan... Mdspans>
constexpr bool all_inner_unit_stride(const Mdspans&... mdspans)
{
  return (inner_unit_stride(mdspans) && ...);
}

template <typename Ptr>
requires std::is_pointer_v<Ptr> constexpr bool is_aligned_for_vec(std::size_t vec_bytes, Ptr p) noexcept
{
  // vec_bytes should be power-of-two for this fast check.
  return (reinterpret_cast<std::uintptr_t>(p) & (vec_bytes - 1)) == 0;
}

template <core::cpts::mdspan Mdspan>
bool is_aligned_for_vec(std::size_t vec_bytes, const Mdspan& m) noexcept
{
  auto p = m.data_handle();
  return is_aligned_for_vec(vec_bytes, p);
}

template <cpts::mdspan... Mdspans>
constexpr bool all_aligned_for_vec(std::size_t vec_bytes, const Mdspans&... mdspans)
{
  return (is_aligned_for_vec(vec_bytes, mdspans) && ...);
}

enum class AccessType : int {
  Scalar = 1,
  Vec2   = 2,
  Vec4   = 4,
};

struct AccessPolicy {
  AccessType load_policy;
  AccessType store_policy;

  __host__ __device__ [[nodiscard]] int load_width() const { return static_cast<int>(load_policy); }

  __host__ __device__ [[nodiscard]] int store_width() const
  {
    return static_cast<int>(store_policy);
  }
};

template <cpts::mdspan Mdspan>
constexpr AccessPolicy determine_policy_mdspan(const Mdspan& mdspan)
{
  static_assert(!cpts::is_layout_left<Mdspan>, "determine_policy: layout_left is not implemented yet.");

  using layout_type = Mdspan::layout_type;

  bool can_vectorize = false;
  if constexpr (cpts::is_layout_right<Mdspan>) {
    can_vectorize = true;
    // fmt::println("is_layout_right");
  } else if constexpr (cpts::is_layout_stride<Mdspan>) {
    // fmt::println("is_layout_stride");
    const bool unit_stride = inner_unit_stride(mdspan);
    // if (!unit_stride) // fmt::println("no vectorize: innermost stride != 1");
    can_vectorize = unit_stride;
  } else {
    // fmt::println("no vectorize: unsupported layout");
  }

  if (can_vectorize) {
    // fmt::println("can_vectorize");
    constexpr std::size_t elem_size = sizeof(typename Mdspan::element_type);
    const std::size_t vec4_bytes = static_cast<std::size_t>(AccessType::Vec4) * elem_size;
    const std::size_t vec2_bytes = static_cast<std::size_t>(AccessType::Vec2) * elem_size;
    if (is_aligned_for_vec(vec4_bytes, mdspan) && rows_aligned_for_vec(mdspan, vec4_bytes))
      return {
        .load_policy  = AccessType::Vec4,
        .store_policy = AccessType::Vec4,
      };
    if (is_aligned_for_vec(vec2_bytes, mdspan) && rows_aligned_for_vec(mdspan, vec2_bytes))
      return {
        .load_policy  = AccessType::Vec2,
        .store_policy = AccessType::Vec2,
      };
    // if (!rows_aligned_for_vec(mdspan, vec4_bytes))
      // fmt::println("no vectorize: row starts not aligned for vec4");
    // if (!rows_aligned_for_vec(mdspan, vec2_bytes))
      // fmt::println("no vectorize: row starts not aligned for vec2");
    // else
      // fmt::println("no vectorize: base pointer alignment insufficient");
  }

  // fallback to scalar
  return {
    .load_policy  = AccessType::Scalar,
    .store_policy = AccessType::Scalar,
  };
}

template <typename KernelTag, cpts::mdspan... Mdspans>
AccessPolicy determine_policy(const Mdspans&... mdspans)
{
  using traits = kernel_traits<KernelTag>;

  if constexpr (!traits::supports_vectorization)
    return {
      .load_policy  = AccessType::Scalar,
      .store_policy = AccessType::Scalar,
    };

  std::array<AccessPolicy, sizeof...(mdspans)> policies = { determine_policy_mdspan(mdspans)... };

  int min_load  = static_cast<int>(AccessType::Vec4);
  int min_store = static_cast<int>(AccessType::Vec4);
  for (const auto& p : policies) {
    min_load  = std::min(min_load, static_cast<int>(p.load_policy));
    min_store = std::min(min_store, static_cast<int>(p.store_policy));
  }

  return
  {
    .load_policy  = static_cast<AccessType>(min_load),
    .store_policy = static_cast<AccessType>(min_store),
  };
}

}  // namespace fast_deconv::core
