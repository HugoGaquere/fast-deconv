#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/linalg/detail/kronecker.cuh>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::linalg {

template <typename T>
inline void kronecker_async(core::stream_resources& resources,
                            core::span_2d<T> A,
                            core::span_2d<T> B,
                            core::span_2d<T> C)
{
  dim3 block(16, 16);
  dim3 grid(CEIL_DIV(C.extent(1), block.x), CEIL_DIV(C.extent(0), block.y));
  detail::kron_kernel_async<<<grid, block, 0, resources.stream>>>(A, B, C);
}

}  // namespace fast_deconv::linalg
