#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>

namespace fast_deconv::linalg::detail {

template <typename T>
__global__ void kron_kernel_async(core::span_2d<T> A, core::span_2d<T> B, core::span_2d<T> C)
{
  using index_t = core::span_idx_t;

  const auto mp = static_cast<index_t>(C.extent(0));
  const auto nq = static_cast<index_t>(C.extent(1));
  const auto p  = static_cast<index_t>(B.extent(0));
  const auto q  = static_cast<index_t>(B.extent(1));

  const index_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const index_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= mp || col >= nq) return;

  // Decompose C(row, col) -> A(a_row, a_col) and B(b_row, b_col)
  const index_t a_row = row / p;  // 0..A.extent(0)-1
  const index_t b_row = row % p;  // 0..p-1
  const index_t a_col = col / q;  // 0..A.extent(1)-1
  const index_t b_col = col % q;  // 0..q-1

  C(row, col) = A(a_row, a_col) * B(b_row, b_col);
}

}  // namespace fast_deconv::linalg::detail
