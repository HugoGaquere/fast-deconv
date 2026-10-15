#include <algorithm>
#include <cstdint>
#include <fast_deconv/linalg/linalg.hpp>

namespace fast_deconv::kernel {

constexpr float kPiSquared = 9.869604403f;

// Gaussian(sigma) at flat index @p idx of one half-complex plane.
__device__ __forceinline__ float gaussian_at(std::int64_t idx, std::int64_t freq_nrow, std::int64_t freq_ncol,
                                             std::int64_t padded_ncol, float sigma)
{
  const std::int64_t row = idx / freq_ncol;
  const std::int64_t col = idx % freq_ncol;
  const float fy =
      static_cast<float>(row < (freq_nrow + 1) / 2 ? row : row - freq_nrow) / static_cast<float>(freq_nrow);
  const float fx = static_cast<float>(col) / static_cast<float>(padded_ncol);
  return expf(-2.0f * kPiSquared * (fy * fy + fx * fx) * sigma * sigma);
}

__global__ void multiply_with_gaussian_kernel(const linalg::complex_type* input, linalg::complex_type* out,
                                              std::int64_t plane, std::int64_t total, std::int64_t freq_nrow,
                                              std::int64_t freq_ncol, std::int64_t padded_ncol, float sigma, float norm)
{
  for (std::int64_t i = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < total;
       i += std::int64_t{blockDim.x} * gridDim.x) {
    const float g = gaussian_at(i % plane, freq_nrow, freq_ncol, padded_ncol, sigma) * norm;
    out[i] = {input[i].x * g, input[i].y * g};
  }
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::linalg {

void weighted_sum_async(const core::exec_ctx& ctx, const core::span3d<const float> A,
                        const core::span1d<const float> weights, core::span2d<float> out)
{
  // _64: A holds n * w elements, which exceeds INT_MAX well before n alone does.
  const std::int64_t w = weights.extent(0);
  const std::int64_t n = out.mapping().required_span_size();
  const std::int64_t lda = A.mapping().stride(0);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  CHECK_CUBLAS(cublasSgemv_64(ctx.cublas_handle, CUBLAS_OP_N,
                              n,                              // m: rows of A_cb
                              w,                              // n: cols of A_cb
                              &alpha, A.data_handle(), lda,   // A, lda
                              weights.data_handle(), 1,       // x, incx
                              &beta, out.data_handle(), 1));  // y, incy
}

void multiply_with_gaussian(const core::exec_ctx& ctx, const fft_dims& dims, std::int64_t n_batch,
                            const complex_type* input, complex_type* out, float sigma)
{
  const std::int64_t plane = dims.freq_total();
  const std::int64_t total = plane * n_batch;
  const auto grid = core::narrow<unsigned>(std::min<std::int64_t>(core::ceil_div(total, 256), 65535));
  kernel::multiply_with_gaussian_kernel<<<grid, 256, 0, ctx.cuda_stream>>>(
      input, out, plane, total, dims.freq_nrow, dims.freq_ncol, dims.padded_ncol, sigma,
      1.0f / static_cast<float>(dims.padded_total()));
}

}  // namespace fast_deconv::linalg
