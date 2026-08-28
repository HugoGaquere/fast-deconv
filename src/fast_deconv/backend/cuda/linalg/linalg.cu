#include <fast_deconv/linalg/linalg.hpp>

namespace fast_deconv::kernel {
__global__ void weighted_sum_kernel(const float* A, const float* weights, float* out, int w, int n)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    float sum = 0.0f;
    for (int c = 0; c < w; c++) {
      sum += A[c * n + i] * weights[c];
    }
    out[i] = sum;
  }
}
}  // namespace fast_deconv::kernel

namespace fast_deconv::linalg {

void weighted_sum_async(const core::exec_ctx& ctx, const float* A, const float* weights, float* out, int w, int n)
{
  kernel::weighted_sum_kernel<<<CEIL_DIV(n, 256), 256, 0, ctx.cuda_stream>>>(A, weights, out, w, n);
}

void weighted_sum_async(const core::exec_ctx& ctx, const core::span3d<const float> A,
                        const core::span1d<const float> weights, core::span2d<float> out)
{
  // const size_t n = A.extent(1) * A.extent(2);
  // kernel::weighted_sum_kernel<<<CEIL_DIV(n, 256), 256, 0, ctx.cuda_stream>>>(
  //     A.data_handle(), weights.data_handle(), out.data_handle(), weights.size(), n);

  const int w = static_cast<int>(weights.size());
  const int n = static_cast<int>(static_cast<std::int64_t>(A.extent(1)) * A.extent(2));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  CHECK_CUBLAS(cublasSgemv(ctx.cublas_handle, CUBLAS_OP_N,
                           n,                              // m: rows of A_cb
                           w,                              // n: cols of A_cb
                           &alpha, A.data_handle(), n,     // A, lda
                           weights.data_handle(), 1,       // x, incx
                           &beta, out.data_handle(), 1));  // y, incy
}

}  // namespace fast_deconv::linalg
