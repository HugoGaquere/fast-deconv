#pragma once
#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>
#include <vector>

namespace fast_deconv::core {

// Below this the fork costs more than the reduce: a combine over a few thousand cached tiles is one compare each.
inline constexpr std::int64_t kMinParallelReduce = 8192;

/// combine over map(i), i in [0, n). @p combine associative and commutative, @p identity its identity.
template <typename T, typename Map, typename Combine>
T reduce(const exec_ctx& /*ctx*/, std::int64_t n, Map map, Combine combine, T identity)
{
  // Serial outside OpenMP: even a parallel region whose if() is false enters the runtime and builds a team.
  if (n <= kMinParallelReduce) {
    T result = identity;
    for (std::int64_t i = 0; i < n; i++) result = combine(result, map(i));
    return result;
  }

  // One partial per thread, combined in thread order: the same result on every run with the same thread count.
  std::vector<T> partials(narrow<std::size_t>(omp_get_max_threads()), identity);
  // firstprivate: read through the shared frame, acc stays in its stack slot all loop; thread copies keep it in
  // registers.
#pragma omp parallel firstprivate(map, combine, identity)
  {
    T acc = identity;
    // nowait: the region's closing barrier already orders every partial before the combine below.
#pragma omp for schedule(static) nowait
    for (std::int64_t i = 0; i < n; i++) acc = combine(acc, map(i));
    partials[narrow<std::size_t>(omp_get_thread_num())] = acc;
  }
  T result = identity;
  for (const T& p : partials) result = combine(result, p);
  return result;
}

}  // namespace fast_deconv::core
