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

// exp(-2 pi^2 sigma^2 f^2) per index of one axis, f = k / period; @p wrap maps the upper half to negative f.
core::mdcontainer<float, 1> gaussian_axis(const core::exec_ctx& ctx, int n, int period, bool wrap, float sigma)
{
  auto g = ctx.alloc_mdcontainer_async<float>(n);
  float* out = g.data_handle();
  for (int k = 0; k < n; k++) {
    const float f = static_cast<float>(wrap && k >= (n + 1) / 2 ? k - n : k) / period;
    out[k] = std::exp(-2.0f * kPiSquared * f * f * sigma * sigma);
  }
  return g;
}

}  // namespace

// The Gaussian is separable, g(row, col) = gy[row] * gx[col]: two small tables replace a per-element exp.
void multiply_with_gaussian(const core::exec_ctx& ctx, const fft_dims& dims, int n_batch, const complex_type* input,
                            complex_type* out, float sigma)
{
  FD_PROFILE_FN();
  const float norm = 1.0f / static_cast<float>(dims.padded_total());
  const auto gy = gaussian_axis(ctx, dims.freq_nrow, dims.freq_nrow, true, sigma);
  const auto gx = gaussian_axis(ctx, dims.freq_ncol, dims.padded_ncol, false, sigma);
  const float* gy_data = gy.data_handle();
  const float* gx_data = gx.data_handle();
  const int ncol = dims.freq_ncol;
#pragma omp parallel for collapse(2)
  for (int b = 0; b < n_batch; b++) {
    for (int r = 0; r < dims.freq_nrow; r++) {
      const std::int64_t base = (static_cast<std::int64_t>(b) * dims.freq_nrow + r) * ncol;
      const float gy_norm = gy_data[r] * norm;
      for (int c = 0; c < ncol; c++) out[base + c] = input[base + c] * (gy_norm * gx_data[c]);
    }
  }
}

void multiply_with_gaussian_once_and_twice(const core::exec_ctx& ctx, const fft_dims& dims, int n_batch,
                                           const complex_type* input, complex_type* out_conv, complex_type* out_conv2,
                                           float sigma)
{
  FD_PROFILE_FN();
  const float norm = 1.0f / static_cast<float>(dims.padded_total());
  const auto gy = gaussian_axis(ctx, dims.freq_nrow, dims.freq_nrow, true, sigma);
  const auto gx = gaussian_axis(ctx, dims.freq_ncol, dims.padded_ncol, false, sigma);
  const float* gy_data = gy.data_handle();
  const float* gx_data = gx.data_handle();
  const int ncol = dims.freq_ncol;
#pragma omp parallel for collapse(2)
  for (int b = 0; b < n_batch; b++) {
    for (int r = 0; r < dims.freq_nrow; r++) {
      const std::int64_t base = (static_cast<std::int64_t>(b) * dims.freq_nrow + r) * ncol;
      const float gy_r = gy_data[r];
      for (int c = 0; c < ncol; c++) {
        const float g = gy_r * gx_data[c];
        out_conv[base + c] = input[base + c] * (g * norm);
        out_conv2[base + c] = input[base + c] * (g * g * norm);
      }
    }
  }
}

}  // namespace fast_deconv::linalg
