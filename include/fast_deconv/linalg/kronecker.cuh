#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/linalg/detail/kronecker.cuh>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::linalg {

template <typename T>
inline void kronecker_async(
  core::stream_resources& resources, const T* A, const T* B, T* C, uint m, uint n, uint k, uint p)
{
  auto stream        = resources.stream;
  const size_t total = m * k * n * p;
  if (total == 0) return;

  constexpr int threads = 256;
  int blocks            = static_cast<int>(CEIL_DIV(total, threads));
  detail::kron_kernel_async<T><<<blocks, threads, 0, stream>>>(A, B, C, m, n, k, p);
}

}  // namespace fast_deconv::linalg
