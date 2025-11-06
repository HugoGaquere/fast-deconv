#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>

namespace fast_deconv::linalg::detail {

template <typename T>
__global__ void kron_kernel_async(const T* A, const T* B, T* C, uint m, uint n, uint k, uint p)
{
  const uint NP    = n * p;
  const uint MK    = m * k;
  const uint total = MK * NP;

  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    const uint rowC = idx / NP;
    const uint colC = idx % NP;

    const uint r = rowC / k;
    const uint u = rowC % k;
    const uint c = colC / p;
    const uint v = colC % p;

    // Row-major indexing
    const T a = A[r * n + c];
    const T b = B[u * p + v];

    C[idx] = a * b;
  }
}

}  // namespace fast_deconv::linalg::detail
