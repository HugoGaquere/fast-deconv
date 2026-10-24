#pragma once

#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::core {

// Below this the fork costs more than the loop: these are one load and one store per element.
inline constexpr std::int64_t kMinParallelForEach = 8192;

/// f(i) for every i in [0, n).
template <typename F>
void for_each_index(const exec_ctx& /*ctx*/, std::int64_t n, F f)
{
  // Serial outside OpenMP: even a parallel region whose if() is false enters the runtime and builds a team.
  if (n <= kMinParallelForEach) {
    for (std::int64_t i = 0; i < n; i++) f(i);
    return;
  }
#pragma omp parallel for firstprivate(f)
  for (std::int64_t i = 0; i < n; i++) f(i);
}

}  // namespace fast_deconv::core
