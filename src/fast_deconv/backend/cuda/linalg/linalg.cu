#include <cstdint>
#include <fast_deconv/linalg/linalg.hpp>
#include <fd_backend/core/launch.cuh>

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
  const std::int64_t i = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const float g = gaussian_at(i % plane, freq_nrow, freq_ncol, padded_ncol, sigma) * norm;
  out[i] = {input[i].x * g, input[i].y * g};
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::linalg {

void multiply_with_gaussian(const core::exec_ctx& ctx, const fft_dims& dims, std::int64_t n_batch,
                            const complex_type* input, complex_type* out, float sigma)
{
  const std::int64_t plane = dims.freq_total();
  const std::int64_t total = plane * n_batch;
  const core::launch_cfg cfg(total);
  core::launch(ctx, cfg, kernel::multiply_with_gaussian_kernel, input, out, plane, total, dims.freq_nrow,
               dims.freq_ncol, dims.padded_ncol, sigma, 1.0f / static_cast<float>(dims.padded_total()));
}

}  // namespace fast_deconv::linalg
