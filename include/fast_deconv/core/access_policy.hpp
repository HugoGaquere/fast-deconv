#pragma once
#include <emu/cuda/device/mdspan.hpp>
#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/core/kernel_traits.hpp>

namespace fast_deconv::core {

template <core::cpts::mdspan Mdspan>
constexpr bool inner_unit_stride(const Mdspan& m)
{
  if constexpr (cpts::is_layout_right<Mdspan>) {
    return true;
  } else if constexpr (cpts::is_layout_left<Mdspan>) {
    return false;
  } else {
    constexpr int R = Mdspan::rank();
    return m.mapping().stride(R - 1) == 1;
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
  } else if constexpr (cpts::is_layout_stride<Mdspan>) {
    can_vectorize = inner_unit_stride(mdspan);
  }

  if (can_vectorize) {
    if (is_aligned_for_vec(static_cast<int>(AccessType::Vec4), mdspan))
      return {
        .load_policy  = AccessType::Vec4,
        .store_policy = AccessType::Vec4,
      };
    if (is_aligned_for_vec(static_cast<int>(AccessType::Vec2), mdspan))
      return {
        .load_policy  = AccessType::Vec2,
        .store_policy = AccessType::Vec2,
      };
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
