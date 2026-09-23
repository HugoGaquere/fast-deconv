#include <algorithm>
#include <cstdint>
#include <fast_deconv/linalg/linalg.hpp>

namespace fast_deconv::kernel {

constexpr float kPiSquared = 9.869604403f;

__global__ void weighted_sum_kernel(const float* A, const float* weights, float* out, int w, int n)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    float sum = 0.0f;
    // Walked, not indexed: c * n can exceed INT_MAX.
    const float* plane = A + i;
    for (int c = 0; c < w; c++, plane += n) {
      sum += *plane * weights[c];
    }
    out[i] = sum;
  }
}

// Gaussian(sigma) at flat index @p idx of one half-complex plane.
__device__ __forceinline__ float gaussian_at(std::int64_t idx, int freq_nrow, int freq_ncol, int padded_ncol,
                                             float sigma)
{
  const int row = static_cast<int>(idx / freq_ncol);
  const int col = static_cast<int>(idx % freq_ncol);
  const float fy = static_cast<float>(row < (freq_nrow + 1) / 2 ? row : row - freq_nrow) / freq_nrow;
  const float fx = static_cast<float>(col) / padded_ncol;
  return expf(-2.0f * kPiSquared * (fy * fy + fx * fx) * sigma * sigma);
}

__global__ void multiply_with_gaussian_kernel(const linalg::complex_type* input, linalg::complex_type* out,
                                              std::int64_t plane, std::int64_t total, int freq_nrow, int freq_ncol,
                                              int padded_ncol, float sigma, float norm)
{
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const float g = gaussian_at(i % plane, freq_nrow, freq_ncol, padded_ncol, sigma) * norm;
    out[i] = {input[i].x * g, input[i].y * g};
  }
}

__global__ void multiply_with_gaussian_once_and_twice_kernel(const linalg::complex_type* input,
                                                             linalg::complex_type* out_conv,
                                                             linalg::complex_type* out_conv2, std::int64_t plane,
                                                             std::int64_t total, int freq_nrow, int freq_ncol,
                                                             int padded_ncol, float sigma, float norm)
{
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const float g = gaussian_at(i % plane, freq_nrow, freq_ncol, padded_ncol, sigma);
    const float g_norm = g * norm;
    const float g2_norm = g * g_norm;
    const linalg::complex_type v = input[i];
    out_conv[i] = {v.x * g_norm, v.y * g_norm};
    out_conv2[i] = {v.x * g2_norm, v.y * g2_norm};
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

  // _64: A holds n * w elements, which exceeds INT_MAX well before n alone does.
  const std::int64_t w = static_cast<std::int64_t>(weights.size());
  const std::int64_t n = static_cast<std::int64_t>(A.extent(1)) * A.extent(2);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  CHECK_CUBLAS(cublasSgemv_64(ctx.cublas_handle, CUBLAS_OP_N,
                              n,                              // m: rows of A_cb
                              w,                              // n: cols of A_cb
                              &alpha, A.data_handle(), n,     // A, lda
                              weights.data_handle(), 1,       // x, incx
                              &beta, out.data_handle(), 1));  // y, incy
}

void multiply_with_gaussian(const core::exec_ctx& ctx, const fft_dims& dims, int n_batch, const complex_type* input,
                            complex_type* out, float sigma)
{
  const std::int64_t plane = dims.freq_total();
  const std::int64_t total = plane * n_batch;
  const int grid = static_cast<int>(std::min<std::int64_t>(CEIL_DIV(total, 256), 65535));
  kernel::multiply_with_gaussian_kernel<<<grid, 256, 0, ctx.cuda_stream>>>(
      input, out, plane, total, dims.freq_nrow, dims.freq_ncol, dims.padded_ncol, sigma,
      1.0f / static_cast<float>(dims.padded_total()));
}

void multiply_with_gaussian_once_and_twice(const core::exec_ctx& ctx, const fft_dims& dims, int n_batch,
                                           const complex_type* input, complex_type* out_conv, complex_type* out_conv2,
                                           float sigma)
{
  const std::int64_t plane = dims.freq_total();
  const std::int64_t total = plane * n_batch;
  const int grid = static_cast<int>(std::min<std::int64_t>(CEIL_DIV(total, 256), 65535));
  kernel::multiply_with_gaussian_once_and_twice_kernel<<<grid, 256, 0, ctx.cuda_stream>>>(
      input, out_conv, out_conv2, plane, total, dims.freq_nrow, dims.freq_ncol, dims.padded_ncol, sigma,
      1.0f / static_cast<float>(dims.padded_total()));
}

}  // namespace fast_deconv::linalg
