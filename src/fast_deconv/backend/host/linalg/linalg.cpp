#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/linalg.hpp>

#include "fast_deconv/core/exec_ctx.hpp"

namespace {
constexpr float kPiSquared = 9.869604403f;
}  // namespace

namespace fast_deconv::linalg {

void weighted_sum_async(const core::exec_ctx& ctx, const float* __restrict A, const float* __restrict weights,
                        float* __restrict out, int w, int n)
{
  FD_PROFILE_FN();
  // Pixel-major: each thread owns a slice of `out`, and the accumulator stays in a register.
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    float acc = 0.0f;
    for (int c = 0; c < w; c++) {
      // Widened before the multiply: w * n can exceed INT_MAX.
      acc += A[static_cast<std::ptrdiff_t>(c) * n + i] * weights[c];
    }
    out[i] = acc;
  }
}

void weighted_sum_async(const core::exec_ctx& ctx, const core::span3d<const float> A,
                        const core::span1d<const float> weights, core::span2d<float> out)
{
  FD_PROFILE_FN();
  // TODO: assert that A.extent(0) == weights.size()
  weighted_sum_async(ctx, A.data_handle(), weights.data_handle(), out.data_handle(), static_cast<int>(weights.size()),
                     static_cast<int>(static_cast<std::int64_t>(A.extent(1)) * A.extent(2)));
}

namespace {

// Gaussian(sigma) at flat index @p idx of one half-complex plane of @p dims.
float gaussian_at(const fft_dims& dims, std::int64_t idx, float sigma)
{
  const int row = static_cast<int>(idx / dims.freq_ncol);
  const int col = static_cast<int>(idx % dims.freq_ncol);
  const int rows = dims.freq_nrow;
  const float fy = static_cast<float>(row < (rows + 1) / 2 ? row : row - rows) / rows;
  const float fx = static_cast<float>(col) / dims.padded_ncol;
  return std::exp(-2.0f * kPiSquared * (fy * fy + fx * fx) * sigma * sigma);
}

}  // namespace

void multiply_with_gaussian(const core::exec_ctx& ctx, const fft_dims& dims, int n_batch, const complex_type* input,
                            complex_type* out, float sigma)
{
  FD_PROFILE_FN();
  const std::int64_t plane = dims.freq_total();
  const std::int64_t total = plane * n_batch;
  const float norm = 1.0f / static_cast<float>(dims.padded_total());
#pragma omp parallel for
  for (std::int64_t i = 0; i < total; i++) out[i] = input[i] * (gaussian_at(dims, i % plane, sigma) * norm);
}

void multiply_with_gaussian_once_and_twice(const core::exec_ctx& ctx, const fft_dims& dims, int n_batch,
                                           const complex_type* input, complex_type* out_conv, complex_type* out_conv2,
                                           float sigma)
{
  FD_PROFILE_FN();
  const std::int64_t plane = dims.freq_total();
  const std::int64_t total = plane * n_batch;
  const float norm = 1.0f / static_cast<float>(dims.padded_total());
#pragma omp parallel for
  for (std::int64_t i = 0; i < total; i++) {
    const float g = gaussian_at(dims, i % plane, sigma);
    out_conv[i] = input[i] * (g * norm);
    out_conv2[i] = input[i] * (g * g * norm);
  }
}

}  // namespace fast_deconv::linalg
