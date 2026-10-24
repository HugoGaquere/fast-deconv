#pragma once
#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>
#include <vector>

namespace fast_deconv::core {

/// combine over map(i), i in [0, n). @p combine associative and commutative, @p identity its identity.
template <typename T, typename Map, typename Combine>
T reduce(const exec_ctx& /*ctx*/, std::int64_t n, Map map, Combine combine, T identity)
{
  // One partial per thread, combined in thread order: the same result on every run with the same thread count.
  std::vector<T> partials(narrow<std::size_t>(omp_get_max_threads()), identity);
#pragma omp parallel
  {
    T acc = identity;
#pragma omp for schedule(static)
    for (std::int64_t i = 0; i < n; i++) acc = combine(acc, map(i));
    partials[narrow<std::size_t>(omp_get_thread_num())] = acc;
  }
  T result = identity;
  for (const T& p : partials) result = combine(result, p);
  return result;
}

}  // namespace fast_deconv::core
