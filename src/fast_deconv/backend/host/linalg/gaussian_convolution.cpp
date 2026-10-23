#include <ducc0/fft/fft.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/gaussian_convolution.hpp>
#include <numbers>

#include "fast_deconv/linalg/fft_dims.hpp"

namespace fast_deconv::linalg {

namespace {
// Image-domain kernel whose DFT is exp(-2 pi^2 sigma^2 (k/n)^2): a circular convolution with it is the
// frequency-domain Gaussian multiply, the inverse transform's 1/n included.
core::mdcontainer<float, 1> gaussian_kernel(const core::exec_ctx& ctx, std::int64_t n, float sigma)
{
  constexpr float kPiSquared = std::numbers::pi_v<float> * std::numbers::pi_v<float>;
  const std::int64_t half_n = n / 2 + 1;
  auto half = ctx.alloc_mdcontainer_async<complex_type>(half_n);
  for (std::int64_t k = 0; k < half_n; k++) {
    const float f = static_cast<float>(k) / static_cast<float>(n);
    half.data_handle()[k] = std::exp(-2.0f * kPiSquared * f * f * sigma * sigma);
  }
  auto kernel = ctx.alloc_mdcontainer_async<float>(n);
  ducc0::c2r(ducc0::cfmav<complex_type>(half.data_handle(), {core::narrow<std::size_t>(half_n)}),
             ducc0::vfmav<float>(kernel.data_handle(), {core::narrow<std::size_t>(n)}), std::size_t{0}, ducc0::BACKWARD,
             1.0f / static_cast<float>(n), 1);
  return kernel;
}

// Lines per batch in both passes: two of ducc0's 8-line FFT bunches, and one 64 B cache line per strip row in the
// column pass.
constexpr std::int64_t kBatchLines = 16;

// Zero-padded convolution of the h x w block at src (row stride ld) through buf: the block goes to the top-left of
// a padded_h x padded_w buffer, the rest is zeroed, the buffer is convolved along axis, and the h x w part is
// copied to dst (row stride ld). src and dst may be the same block.
void convolve_block(const float* src, float* dst, std::int64_t ld, std::int64_t h, std::int64_t w,
                    std::int64_t padded_h, std::int64_t padded_w, std::size_t axis, const ducc0::cmav<float, 1>& kernel,
                    float* buf)
{
  // A full column strip: kBatchLines floats per row and no row padding. The constant width lets the compiler
  // inline each row copy as two 32 B loads and stores instead of a memmove call per 64 B row.
  const bool full_strip = w == kBatchLines && padded_w == kBatchLines;
  if (full_strip) {
    for (std::int64_t i = 0; i < h; i++) std::copy_n(src + i * ld, kBatchLines, buf + i * kBatchLines);
  } else {
    for (std::int64_t i = 0; i < h; i++) {
      std::copy_n(src + i * ld, w, buf + i * padded_w);
      std::fill(buf + i * padded_w + w, buf + (i + 1) * padded_w, 0.0f);
    }
  }
  std::fill(buf + h * padded_w, buf + padded_h * padded_w, 0.0f);
  const ducc0::vfmav<float> lines(buf, {core::narrow<std::size_t>(padded_h), core::narrow<std::size_t>(padded_w)});
  ducc0::convolve_axis(lines, lines, axis, kernel, 1);
  if (full_strip) {
    for (std::int64_t i = 0; i < h; i++) std::copy_n(buf + i * kBatchLines, kBatchLines, dst + i * ld);
  } else {
    for (std::int64_t i = 0; i < h; i++) std::copy_n(buf + i * padded_w, w, dst + i * ld);
  }
}

}  // namespace

gaussian_convolution_ctx::gaussian_convolution_ctx(const core::exec_ctx& ctx, std::int64_t batch, std::int64_t nrow,
                                                   std::int64_t ncol, const std::vector<float>&, float)
    : ctx_(ctx), batch_(batch), nrow_(nrow), ncol_(ncol)
{
}

gaussian_convolution_ctx::spectrum gaussian_convolution_ctx::make_spectrum() const
{
  return ctx_.alloc_mdcontainer_async<float>(batch_, nrow_, ncol_);
}

void gaussian_convolution_ctx::forward(core::span3d<const float> input, spectrum& out) const
{
  FD_PROFILE_FN();
  assert(input.is_exhaustive() && input.extent(0) == batch_);
  assert(input.extent(1) == nrow_ && input.extent(2) == ncol_);
  assert(out.extent(0) == batch_ && out.extent(1) == nrow_ && out.extent(2) == ncol_);

  const float* in = input.data_handle();
  float* copy = out.data_handle();
  const std::int64_t n_rows = batch_ * nrow_;
  const std::int64_t ncol = ncol_;
  // Row r of the (batch * nrow, ncol) stack is the ncol floats starting at r * ncol, in both buffers.
#pragma omp parallel for
  for (std::int64_t r = 0; r < n_rows; r++) std::copy(in + r * ncol, in + (r + 1) * ncol, copy + r * ncol);
}

void gaussian_convolution_ctx::convolve(const spectrum& in, float sigma, core::span3d<float> out) const
{
  FD_PROFILE_FN();
  assert(in.extent(0) == batch_ && in.extent(1) == nrow_ && in.extent(2) == ncol_);
  convolve_separable_(in.data_handle(), sigma, out);
}

// Straight from the input: the copy into a spectrum is only needed to reuse it across sigmas.
void gaussian_convolution_ctx::convolve(core::span3d<const float> input, float sigma, core::span3d<float> out) const
{
  FD_PROFILE_FN();
  assert(input.is_exhaustive() && input.extent(0) == batch_);
  assert(input.extent(1) == nrow_ && input.extent(2) == ncol_);
  convolve_separable_(input.data_handle(), sigma, out);
}

// Separable Gaussian through per-thread line batches. Each line (row, then column) is zero-padded on the fly in a
// small per-thread buffer, convolved there by ducc0::convolve_axis, and its image part written to `out`; the
// padding never exists as a plane. The row pass convolves along the buffer's contiguous axis. The column pass
// copies kBatchLines adjacent columns, one 64 B cache line per row, into a buffer with the strip's own layout and
// convolves along its axis 0: one column per SIMD lane, as ducc0's FFT bunches want, so nothing is transposed.
// The padding is per sigma, from the kernel's reach.
void gaussian_convolution_ctx::convolve_separable_(const float* in, float sigma, core::span3d<float> out) const
{
  assert(out.is_exhaustive() && out.extent(0) == batch_);
  assert(out.extent(1) == nrow_ && out.extent(2) == ncol_);

  const std::int64_t nrow = nrow_;
  const std::int64_t ncol = ncol_;
  // P - n >= reach: a wrapped tail crosses the zero strip, in either direction, before it can reach the image.
  const std::int64_t pad = gaussian_reach(sigma);
  const std::int64_t row_len = next_fast_size(ncol + pad);
  const std::int64_t col_len = next_fast_size(nrow + pad);

  const auto kx = gaussian_kernel(ctx_, row_len, sigma);
  const auto ky = gaussian_kernel(ctx_, col_len, sigma);
  const ducc0::cmav<float, 1> kernel_x(kx.data_handle(), {core::narrow<std::size_t>(row_len)});
  const ducc0::cmav<float, 1> kernel_y(ky.data_handle(), {core::narrow<std::size_t>(col_len)});

  float* dst = out.data_handle();
  const std::int64_t scratch_size_per_thread = kBatchLines * std::max(row_len, col_len);
  const auto scratch = ctx_.alloc_ptr_async<float>(omp_get_max_threads() * scratch_size_per_thread);

  // Rows: all batch planes are one stack of batch * nrow contiguous rows.
  {
    FD_PROFILE_SCOPE("convolve/rows");
    const std::int64_t n_rows = batch_ * nrow;
    const std::int64_t n_blocks = core::ceil_div(n_rows, kBatchLines);
#pragma omp parallel
    {
      float* buf = scratch.get() + omp_get_thread_num() * scratch_size_per_thread;
#pragma omp for schedule(static)
      for (std::int64_t blk = 0; blk < n_blocks; blk++) {
        // Stack rows r0 .. r0 + rb - 1, each zero-padded to row_len; the last block may hold fewer than kBatchLines.
        const std::int64_t r0 = blk * kBatchLines;
        const std::int64_t rb = std::min(kBatchLines, n_rows - r0);
        convolve_block(in + r0 * ncol, dst + r0 * ncol, ncol, rb, ncol, rb, row_len, 1, kernel_x, buf);
      }
    }
  }
  // Columns, in place in `out`: kBatchLines adjacent columns of one plane at a time, each zero-padded to col_len.
  {
    FD_PROFILE_SCOPE("convolve/cols");
    const std::int64_t blocks_per_plane = core::ceil_div(ncol, kBatchLines);
    const std::int64_t n_blocks = batch_ * blocks_per_plane;
#pragma omp parallel
    {
      float* buf = scratch.get() + omp_get_thread_num() * scratch_size_per_thread;
#pragma omp for schedule(static)
      for (std::int64_t blk = 0; blk < n_blocks; blk++) {
        // Block blk = plane blk / blocks_per_plane, columns c0 .. c0 + cb - 1 of that plane.
        float* plane = dst + (blk / blocks_per_plane) * nrow * ncol;
        const std::int64_t c0 = (blk % blocks_per_plane) * kBatchLines;
        const std::int64_t cb = std::min(kBatchLines, ncol - c0);
        convolve_block(plane + c0, plane + c0, ncol, nrow, cb, col_len, cb, 0, kernel_y, buf);
      }
    }
  }
}

}  // namespace fast_deconv::linalg
