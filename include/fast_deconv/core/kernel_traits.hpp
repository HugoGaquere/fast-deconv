#pragma once

#include <cstdint>

namespace fast_deconv::core {

template <typename KernelTag>
struct kernel_traits;

enum class BoundType { Memory, Compute, Balanced };

struct subtract_kernel_tag {};

template <>
struct kernel_traits<subtract_kernel_tag> {
  static constexpr BoundType bound_type          = BoundType::Memory;
  static constexpr bool supports_vectorization   = true;
  static constexpr uint32_t preferred_block_size = 256;
};

}  // namespace fast_deconv::core
