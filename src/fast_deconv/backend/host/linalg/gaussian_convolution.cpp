#include <ducc0/fft/fft.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/gaussian_convolution.hpp>
#include <numbers>

#include "fast_deconv/linalg/fft_dims.hpp"

namespace fast_deconv::linalg {

namespace {
// Image-domain kernel whose DFT is exp(-2 pi^2 sigma^2 (k/n)^2): a circular convolution with it is the
// frequency-domain Gaussian multiply, the inverse transform's 1/n included.
core::mdcontainer<float, 1> gaussian_kernel(const core::exec_ctx& ctx, int n, float sigma)
{
  constexpr float kPiSquared = std::numbers::pi_v<float> * std::numbers::pi_v<float>;
  const auto half_n = static_cast<std::size_t>(n / 2 + 1);
  auto half = ctx.alloc_mdcontainer_async<complex_type>(half_n);
  for (std::size_t k = 0; k < half_n; k++) {
    const float f = static_cast<float>(k) / static_cast<float>(n);
    half.data_handle()[k] = std::exp(-2.0f * kPiSquared * f * f * sigma * sigma);
  }
  auto kernel = ctx.alloc_mdcontainer_async<float>(n);
  ducc0::c2r(ducc0::cfmav<complex_type>(half.data_handle(), {half_n}),
             ducc0::vfmav<float>(kernel.data_handle(), {static_cast<std::size_t>(n)}), std::size_t{0}, ducc0::BACKWARD,
             1.0f / static_cast<float>(n), 1);
  return kernel;
}
// Lines per tile in the tiled Gaussian: 16 floats are one 64 B cache line, so the column gather reads whole
// lines, and a multiple of ducc0's SIMD bunch (4 lines on SSE2, 8 on AVX2).
constexpr std::size_t kTileLines = 16;

// Last pixel where the image-domain kernel of exp(-2 pi^2 sigma^2 f^2), f in [-1/2, 1/2], is above
// kReachTol of its peak: float32 precision. Closed form, an upper bound on the measured reach (within
// 1-3 px for sigma in [0.05, 200], see gaussian_conv/padding_and_kernel_reach.ipynb):
//   - the Gaussian decay: exp(-x^2 / 2 sigma^2) = tol  =>  x = sigma sqrt(2 ln(1/tol));
//   - the cut at Nyquist (small sigma): |h(x)| ~ sigma^2 H(1/2) / x^2, against the peak
//     h(0) = erf(pi sigma / sqrt 2) / (sigma sqrt(2 pi)).
int gaussian_reach(float sigma)
{
  constexpr double kReachTol = 1e-7;
  if (!(sigma > 0.0f)) return 0;  // Gaussian(0) is the identity
  const double s = sigma;
  const double pi = std::numbers::pi;
  const double gaussian = s * std::sqrt(2.0 * std::log(1.0 / kReachTol));
  const double peak = std::erf(pi * s / std::sqrt(2.0)) / (s * std::sqrt(2.0 * pi));
  const double h_nyquist = std::exp(-pi * pi * s * s / 2.0);
  const double tail = std::sqrt(s * s * h_nyquist / (kReachTol * peak));
  return static_cast<int>(std::ceil(std::max(gaussian, tail)));
}
}  // namespace

gaussian_convolution_ctx::gaussian_convolution_ctx(const core::exec_ctx& ctx, int batch, int nrow, int ncol,
                                                   float /*padding*/)
    : ctx_(ctx), batch_(batch), nrow_(nrow), ncol_(ncol)
{
}

gaussian_convolution_ctx::spectrum gaussian_convolution_ctx::make_spectrum() const
{
  return ctx_.alloc_mdcontainer_async<float>(batch_, nrow_, ncol_);
}

// On this backend the spectrum is a copy of the input: convolve() needs no transform, only the data.
void gaussian_convolution_ctx::forward(core::span3d<const float> input, spectrum& out) const
{
  FD_PROFILE_FN();
  assert(input.is_exhaustive() && input.extent(0) == batch_);
  assert(input.extent(1) == nrow_ && input.extent(2) == ncol_);
  assert(out.extent(0) == batch_ && out.extent(1) == nrow_ && out.extent(2) == ncol_);

  const float* in = input.data_handle();
  float* copy = out.data_handle();
  const std::ptrdiff_t n_rows = static_cast<std::ptrdiff_t>(batch_) * nrow_;
  const std::ptrdiff_t ncol = ncol_;
  // Row r of the (batch * nrow, ncol) stack is the ncol floats starting at r * ncol, in both buffers.
#pragma omp parallel for
  for (std::ptrdiff_t r = 0; r < n_rows; r++) std::copy(in + r * ncol, in + (r + 1) * ncol, copy + r * ncol);
}

void gaussian_convolution_ctx::convolve(const spectrum& in, float sigma, core::span3d<float> out) const
{
  FD_PROFILE_FN();
  assert(in.extent(0) == batch_ && in.extent(1) == nrow_ && in.extent(2) == ncol_);
  convolve_tiled_(in.data_handle(), sigma, out);
}

// Straight from the input: the copy into a spectrum is only needed to reuse it across sigmas.
void gaussian_convolution_ctx::convolve(core::span3d<const float> input, float sigma, core::span3d<float> out) const
{
  FD_PROFILE_FN();
  assert(input.is_exhaustive() && input.extent(0) == batch_);
  assert(input.extent(1) == nrow_ && input.extent(2) == ncol_);
  convolve_tiled_(input.data_handle(), sigma, out);
}

// Separable Gaussian through per-thread tiles. Each line (row, then column) is zero-padded on the fly in a
// small per-thread buffer, convolved there by ducc0::convolve_axis along the contiguous axis, and its image
// part written to `out`; the padding never exists as a plane. The column pass gathers kTileLines adjacent
// columns, one 64 B cache line per row, transposed into the buffer. The padding is per sigma, from the
// kernel's reach (gaussian_reach).
void gaussian_convolution_ctx::convolve_tiled_(const float* in, float sigma, core::span3d<float> out) const
{
  assert(out.is_exhaustive() && out.extent(0) == batch_);
  assert(out.extent(1) == nrow_ && out.extent(2) == ncol_);

  const auto nrow = static_cast<std::size_t>(nrow_);
  const auto ncol = static_cast<std::size_t>(ncol_);
  const auto batch = static_cast<std::size_t>(batch_);
  // P - n >= reach: a wrapped tail crosses the zero strip, in either direction, before it can reach the image.
  const int reach = gaussian_reach(sigma);
  const auto row_len = static_cast<std::size_t>(next_fast_size(ncol_ + reach));  // padded row length
  const auto col_len = static_cast<std::size_t>(next_fast_size(nrow_ + reach));  // padded column length

  const auto kx = gaussian_kernel(ctx_, static_cast<int>(row_len), sigma);
  const auto ky = gaussian_kernel(ctx_, static_cast<int>(col_len), sigma);
  const ducc0::cmav<float, 1> kernel_x(kx.data_handle(), {row_len});
  const ducc0::cmav<float, 1> kernel_y(ky.data_handle(), {col_len});

  float* dst = out.data_handle();
  const std::size_t tile = kTileLines * std::max(row_len, col_len);
  const auto scratch = ctx_.alloc_ptr_async<float>(static_cast<std::size_t>(omp_get_max_threads()) * tile);

  // Rows: all batch planes are one stack of batch * nrow contiguous rows.
  {
    FD_PROFILE_SCOPE("convolve/rows");
    const auto n_rows = batch * nrow;
    const auto n_blocks = static_cast<std::ptrdiff_t>((n_rows + kTileLines - 1) / kTileLines);
#pragma omp parallel
    {
      float* buf = scratch.get() + static_cast<std::size_t>(omp_get_thread_num()) * tile;
#pragma omp for schedule(static)
      for (std::ptrdiff_t blk = 0; blk < n_blocks; blk++) {
        // This block is stack rows r0 .. r0 + rb - 1; the last block may hold fewer than kTileLines.
        const std::size_t r0 = static_cast<std::size_t>(blk) * kTileLines;
        const std::size_t rb = std::min(kTileLines, n_rows - r0);
        // Tile line r = stack row r0 + r, then zeros up to row_len.
        for (std::size_t r = 0; r < rb; r++) {
          float* line = buf + r * row_len;
          std::copy(in + (r0 + r) * ncol, in + (r0 + r + 1) * ncol, line);
          std::fill(line + ncol, line + row_len, 0.0f);
        }
        const ducc0::vfmav<float> lines(buf, {rb, row_len});
        ducc0::convolve_axis(lines, lines, std::size_t{1}, kernel_x, 1);
        // Crop: the first ncol floats of each tile line go back to their stack row in `out`.
        for (std::size_t r = 0; r < rb; r++)
          std::copy(buf + r * row_len, buf + r * row_len + ncol, dst + (r0 + r) * ncol);
      }
    }
  }
  // Columns, in place in `out`: kTileLines adjacent columns of one plane at a time, gathered transposed.
  {
    FD_PROFILE_SCOPE("convolve/cols");
    const std::size_t blocks_per_plane = (ncol + kTileLines - 1) / kTileLines;
    const auto n_blocks = static_cast<std::ptrdiff_t>(batch * blocks_per_plane);
#pragma omp parallel
    {
      float* buf = scratch.get() + static_cast<std::size_t>(omp_get_thread_num()) * tile;
#pragma omp for schedule(static)
      for (std::ptrdiff_t blk = 0; blk < n_blocks; blk++) {
        // Block blk = plane blk / blocks_per_plane, columns c0 .. c0 + cb - 1 of that plane.
        float* plane = dst + (static_cast<std::size_t>(blk) / blocks_per_plane) * nrow * ncol;
        const std::size_t c0 = (static_cast<std::size_t>(blk) % blocks_per_plane) * kTileLines;
        const std::size_t cb = std::min(kTileLines, ncol - c0);
        // Tile line c holds column c0 + c: image rows in [0, nrow), zeros in [nrow, col_len).
        for (std::size_t c = 0; c < cb; c++) std::fill(buf + c * col_len + nrow, buf + (c + 1) * col_len, 0.0f);
        // Gather (transpose): pixel (i, c0 + c) goes to tile line c, slot i.
        for (std::size_t i = 0; i < nrow; i++) {
          const float* src = plane + i * ncol + c0;
          for (std::size_t c = 0; c < cb; c++) buf[c * col_len + i] = src[c];
        }
        const ducc0::vfmav<float> lines(buf, {cb, col_len});
        ducc0::convolve_axis(lines, lines, std::size_t{1}, kernel_y, 1);
        // Scatter (the same transpose back): tile line c, slot i returns to pixel (i, c0 + c).
        for (std::size_t i = 0; i < nrow; i++) {
          float* row = plane + i * ncol + c0;
          for (std::size_t c = 0; c < cb; c++) row[c] = buf[c * col_len + i];
        }
      }
    }
  }
}

}  // namespace fast_deconv::linalg
