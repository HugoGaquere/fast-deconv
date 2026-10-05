#include <ducc0/fft/fft.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/convolution.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <numbers>
#include <utility>

#include "fast_deconv/linalg/fft_dims.hpp"

namespace fast_deconv::linalg {

namespace {
// (n_batch, nrow, ncol) row-major; ducc0 derives contiguous strides from the shape.
// ducc0::fmav_info::shape_t cube_shape(int n_batch, int nrow, int ncol)
// {
//   return {static_cast<std::size_t>(n_batch), static_cast<std::size_t>(nrow), static_cast<std::size_t>(ncol)};
// }

// Axis 0 is the batch and stays untransformed; the r2c/c2r axis must come last.
// const ducc0::fmav_info::shape_t kFftAxes{1, 2};

// ducc0 runs its own pool, capped at hardware concurrency, not the OpenMP one.
// std::size_t fft_threads() { return static_cast<std::size_t>(omp_get_max_threads()); }

// Column blocks in multiples of ducc0's widest strided bunch, so no two threads share a cache line.
constexpr std::size_t kColumnAlign = 16;

// The calling OpenMP thread's contiguous share of [0, n), boundaries on multiples of align.
std::pair<std::size_t, std::size_t> thread_share(std::size_t n, std::size_t align)
{
  const std::size_t units = (n + align - 1) / align;
  const auto t = static_cast<std::size_t>(omp_get_thread_num());
  const auto nt = static_cast<std::size_t>(omp_get_num_threads());
  return {std::min(units * t / nt * align, n), std::min(units * (t + 1) / nt * align, n)};
}

// In-place c2c down the columns (axis 1): each thread takes a column block, all rows of every batch plane.
// nthreads = 1 makes ducc0 run inline on the OpenMP thread, so OMP_PROC_BIND places the FFT too.
void columns_c2c(complex_type* data, int n_batch, int nrow, int ncol, bool forward)
{
#pragma omp parallel
  {
    const auto [c0, c1] = thread_share(static_cast<std::size_t>(ncol), kColumnAlign);
    if (c0 < c1) {
      const ducc0::vfmav<complex_type> cols(
          data + c0, {static_cast<std::size_t>(n_batch), static_cast<std::size_t>(nrow), c1 - c0},
          {static_cast<std::ptrdiff_t>(nrow) * ncol, ncol, 1});
      ducc0::c2c(cols, cols, {1}, forward, 1.0f, 1);
    }
  }
}

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

// TODO: old signature
void pad_ifftshift_async(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output)
{
  FD_PROFILE_FN();
  // Parallel fill: the padded plane is gigabytes here, and this first-touches it.
  const int padded_total = dims.padded_total();
#pragma omp parallel for
  for (int i = 0; i < padded_total; i++) output[i] = 0.0f;

  // The (row, col) -> (out_row, out_col) map is a bijection, so writes never collide.
#pragma omp parallel for
  for (int i = 0; i < dims.input_nrow; i++) {
    for (int j = 0; j < dims.input_ncol; j++) {
      // Position in padded array (input centered)
      const int pad_row = i + dims.padding_nrow;
      const int pad_col = j + dims.padding_ncol;

      // ifftshift: shift by ceil(N/2) = (N+1)/2
      const int out_row = (pad_row + (dims.padded_nrow + 1) / 2) % dims.padded_nrow;
      const int out_col = (pad_col + (dims.padded_ncol + 1) / 2) % dims.padded_ncol;

      output[out_row * dims.padded_ncol + out_col] = input[i * dims.input_ncol + j];
    }
  }
}

void pad_ifftshift_batched_async(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output,
                                 int n_batch)
{
  FD_PROFILE_FN();
  for (int b = 0; b < n_batch; b++) {
    // Widened before the multiply: a batch of planes can exceed INT_MAX elements.
    const std::ptrdiff_t in_off = static_cast<std::ptrdiff_t>(b) * dims.input_total();
    const std::ptrdiff_t out_off = static_cast<std::ptrdiff_t>(b) * dims.padded_total();
    pad_ifftshift_async(ctx, dims, input + in_off, output + out_off);
  }
}

void fftshift_crop_async(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output,
                         int n_batch)
{
  FD_PROFILE_FN();
  // Collapsed: n_batch alone is a handful of slices, too few to fill the pool.
#pragma omp parallel for collapse(2)
  for (int b = 0; b < n_batch; b++) {
    for (int i = 0; i < dims.input_nrow; i++) {
      const float* in = input + static_cast<std::ptrdiff_t>(b) * dims.padded_total();
      float* out = output + static_cast<std::ptrdiff_t>(b) * dims.input_total();
      const int src_row = (i + dims.padding_nrow + (dims.padded_nrow + 1) / 2) % dims.padded_nrow;

      for (int j = 0; j < dims.input_ncol; j++) {
        const int src_col = (j + dims.padding_ncol + (dims.padded_ncol + 1) / 2) % dims.padded_ncol;
        out[i * dims.input_ncol + j] = in[src_row * dims.padded_ncol + src_col];
      }
    }
  }
}

convolution_ctx::convolution_ctx(const core::exec_ctx& ctx, int nrow, int ncol, float padding, int batch)
    : ctx_(ctx),
      dims_(nrow, ncol, padding),
      batch_(batch),
      padded_(ctx.alloc_ptr_async<float>(static_cast<std::size_t>(batch) * dims_.padded_total())),
      product_(ctx.alloc_ptr_async<complex_type>(static_cast<std::size_t>(batch) * dims_.freq_total()))
{
}

// Previous version: one ducc0 call per pass on its own thread pool.
// void convolution_ctx::forward_(float* input, complex_type* output) const
// {
//   FD_PROFILE_FN();
//   const ducc0::cfmav<float> in(input, cube_shape(batch_, dims_.padded_nrow, dims_.padded_ncol));
//   const ducc0::vfmav<complex_type> out(output, cube_shape(batch_, dims_.freq_nrow, dims_.freq_ncol));
//   ducc0::r2c(in, out, kFftAxes, ducc0::FORWARD, 1.0f, fft_threads());
// }
//
// void convolution_ctx::backward_(complex_type* input, float* output) const
// {
//   FD_PROFILE_FN();
//   const ducc0::vfmav<complex_type> freq(input, cube_shape(batch_, dims_.freq_nrow, dims_.freq_ncol));
//   const ducc0::vfmav<float> out(output, cube_shape(batch_, dims_.padded_nrow, dims_.padded_ncol));
//   const auto threads = fft_threads();
//   // The row transform runs in place on the input, so ducc0 allocates no intermediate.
//   {
//     FD_PROFILE_SCOPE("backward/c2c");
//     ducc0::c2c(freq, freq, {1}, ducc0::BACKWARD, 1.0f, threads);
//   }
//   {
//     FD_PROFILE_SCOPE("backward/c2r");
//     ducc0::c2r(freq, out, std::size_t{2}, ducc0::BACKWARD, 1.0f, threads);
//   }
// }

// fct = 1 keeps cuFFT's unnormalized convention: the Gaussian multiply applies 1/padded_total.
// Batch and row axes flatten into one: rows are contiguous and freq_nrow == padded_nrow.
void convolution_ctx::forward_(float* input, complex_type* output) const
{
  FD_PROFILE_FN();
  const int n_rows = batch_ * dims_.padded_nrow;
  {
    FD_PROFILE_SCOPE("forward/r2c");
#pragma omp parallel
    {
      const auto [r0, r1] = thread_share(static_cast<std::size_t>(n_rows), 1);
      if (r0 < r1) {
        const ducc0::cfmav<float> in(input + r0 * dims_.padded_ncol,
                                     {r1 - r0, static_cast<std::size_t>(dims_.padded_ncol)});
        const ducc0::vfmav<complex_type> out(output + r0 * dims_.freq_ncol,
                                             {r1 - r0, static_cast<std::size_t>(dims_.freq_ncol)});
        ducc0::r2c(in, out, std::size_t{1}, ducc0::FORWARD, 1.0f, 1);
      }
    }
  }
  {
    FD_PROFILE_SCOPE("forward/c2c");
    columns_c2c(output, batch_, dims_.freq_nrow, dims_.freq_ncol, ducc0::FORWARD);
  }
}

void convolution_ctx::backward_(complex_type* input, float* output) const
{
  FD_PROFILE_FN();
  // The column transform runs in place on the input, so ducc0 allocates no intermediate.
  {
    FD_PROFILE_SCOPE("backward/c2c");
    columns_c2c(input, batch_, dims_.freq_nrow, dims_.freq_ncol, ducc0::BACKWARD);
  }
  {
    FD_PROFILE_SCOPE("backward/c2r");
    const int n_rows = batch_ * dims_.freq_nrow;
#pragma omp parallel
    {
      const auto [r0, r1] = thread_share(static_cast<std::size_t>(n_rows), 1);
      if (r0 < r1) {
        const ducc0::cfmav<complex_type> in(input + r0 * dims_.freq_ncol,
                                            {r1 - r0, static_cast<std::size_t>(dims_.freq_ncol)});
        const ducc0::vfmav<float> out(output + r0 * dims_.padded_ncol,
                                      {r1 - r0, static_cast<std::size_t>(dims_.padded_ncol)});
        ducc0::c2r(in, out, std::size_t{1}, ducc0::BACKWARD, 1.0f, 1);
      }
    }
  }
}

// Previous version: rows then kept columns over a padding-factor-sized P x P plane (padded_), with zero and
// crop passes. The column pass reads 16 B of each 64 B cache line on SSE2 builds (4-float ducc0 bunches).
// // The Gaussian is separable, so each scale is two ducc0::convolve_axis passes over the zero-padded
// // image, each line transformed in cache, instead of multiply + two full inverse passes + crop. On this
// // backend the "spectrum" buffer only carries the column-padded input, (batch, input_nrow, padded_ncol).
// void convolution_ctx::forward(core::span3d<const float> input, core::span1d<complex_type> spectrum) const
// {
//   FD_PROFILE_FN();
//   assert(input.is_exhaustive() && input.extent(0) == batch_);
//   assert(input.extent(1) == dims_.input_nrow && input.extent(2) == dims_.input_ncol);
//   assert(spectrum.size() == static_cast<std::size_t>(batch_) * dims_.freq_total());
//
//   const float* in = input.data_handle();
//   float* rows = reinterpret_cast<float*>(spectrum.data_handle());
//   const std::ptrdiff_t n_rows = static_cast<std::ptrdiff_t>(batch_) * dims_.input_nrow;
// #pragma omp parallel for
//   for (std::ptrdiff_t r = 0; r < n_rows; r++) {
//     float* row = rows + r * dims_.padded_ncol;
//     std::fill(row, row + dims_.padded_ncol, 0.0f);
//     std::copy(in + r * dims_.input_ncol, in + (r + 1) * dims_.input_ncol, row + dims_.padding_ncol);
//   }
// }
//
// void convolution_ctx::convolve_spectrum(core::span1d<const complex_type> spectrum, float sigma,
//                                         core::span3d<float> out) const
// {
//   FD_PROFILE_FN();
//   assert(spectrum.size() == static_cast<std::size_t>(batch_) * dims_.freq_total());
//   assert(out.is_exhaustive() && out.extent(0) == batch_);
//   assert(out.extent(1) == dims_.input_nrow && out.extent(2) == dims_.input_ncol);
//
//   const auto kx = gaussian_kernel(ctx_, dims_.padded_ncol, sigma);
//   const auto ky = gaussian_kernel(ctx_, dims_.padded_nrow, sigma);
//   const ducc0::cmav<float, 1> kernel_x(kx.data_handle(), {static_cast<std::size_t>(dims_.padded_ncol)});
//   const ducc0::cmav<float, 1> kernel_y(ky.data_handle(), {static_cast<std::size_t>(dims_.padded_nrow)});
//
//   const auto nrow = static_cast<std::size_t>(dims_.input_nrow);
//   const auto ncol = static_cast<std::size_t>(dims_.input_ncol);
//   const auto pnrow = static_cast<std::size_t>(dims_.padded_nrow);
//   const auto pncol = static_cast<std::size_t>(dims_.padded_ncol);
//   const auto prow0 = static_cast<std::size_t>(dims_.padding_nrow);
//   const auto pcol0 = static_cast<std::size_t>(dims_.padding_ncol);
//   const auto* rows = reinterpret_cast<const float*>(spectrum.data_handle());
//   float* work = padded_.get();  // (batch, padded_nrow, padded_ncol); only the kept columns are read back
//
//   // padded_ is shared with the FFT path, so the padding rows of the kept columns are re-zeroed each call.
//   {
//     FD_PROFILE_SCOPE("convolve/zero");
// #pragma omp parallel for collapse(2)
//     for (int b = 0; b < batch_; b++) {
//       for (std::size_t r = 0; r < pnrow; r++) {
//         if (r >= prow0 && r < prow0 + nrow) continue;
//         float* row = work + (b * pnrow + r) * pncol + pcol0;
//         std::fill(row, row + ncol, 0.0f);
//       }
//     }
//   }
//   // Rows: only the input rows, since the padding rows are zero and stay zero.
//   {
//     FD_PROFILE_SCOPE("convolve/rows");
// #pragma omp parallel
//     {
//       const auto [r0, r1] = thread_share(nrow, 1);
//       for (int b = 0; b < batch_ && r0 < r1; b++) {
//         const ducc0::cfmav<float> in(rows + (b * nrow + r0) * pncol, {r1 - r0, pncol});
//         const ducc0::vfmav<float> dst(work + (b * pnrow + prow0 + r0) * pncol, {r1 - r0, pncol});
//         ducc0::convolve_axis(in, dst, std::size_t{1}, kernel_x, 1);
//       }
//     }
//   }
//   // Columns: only the kept ones, all padded rows, in place.
//   {
//     FD_PROFILE_SCOPE("convolve/cols");
// #pragma omp parallel
//     {
//       const auto [c0, c1] = thread_share(ncol, kColumnAlign);
//       for (int b = 0; b < batch_ && c0 < c1; b++) {
//         const ducc0::vfmav<float> cols(work + b * pnrow * pncol + pcol0 + c0, {pnrow, c1 - c0},
//                                        {static_cast<std::ptrdiff_t>(pncol), 1});
//         ducc0::convolve_axis(cols, cols, std::size_t{0}, kernel_y, 1);
//       }
//     }
//   }
//   {
//     FD_PROFILE_SCOPE("convolve/crop");
//     float* dst = out.data_handle();
//     const std::ptrdiff_t n_rows = static_cast<std::ptrdiff_t>(batch_) * dims_.input_nrow;
// #pragma omp parallel for
//     for (std::ptrdiff_t r = 0; r < n_rows; r++) {
//       const std::ptrdiff_t b = r / dims_.input_nrow;
//       const std::ptrdiff_t i = r % dims_.input_nrow;
//       const float* src =
//           work + (b * dims_.padded_nrow + dims_.padding_nrow + i) * dims_.padded_ncol + dims_.padding_ncol;
//       std::copy(src, src + dims_.input_ncol, dst + r * dims_.input_ncol);
//     }
//   }
// }

// Separable Gaussian through per-thread tiles. Each line (row, then column) is zero-padded on the fly in a
// small per-thread buffer, convolved there by ducc0::convolve_axis along the contiguous axis, and its image
// part written to `out`; the padding never exists as a plane. The column pass gathers kTileLines adjacent
// columns, one 64 B cache line per row, transposed into the buffer. The padding is per scale, from the
// kernel's reach (gaussian_reach): the fft_padding factor (dims_) no longer applies to this path.
// On this backend the "spectrum" buffer only carries a copy of the input, (batch, input_nrow, input_ncol).
void convolution_ctx::forward(core::span3d<const float> input, core::span1d<complex_type> spectrum) const
{
  FD_PROFILE_FN();
  assert(input.is_exhaustive() && input.extent(0) == batch_);
  assert(input.extent(1) == dims_.input_nrow && input.extent(2) == dims_.input_ncol);
  assert(spectrum.size() == static_cast<std::size_t>(batch_) * dims_.freq_total());

  const float* in = input.data_handle();
  float* copy = reinterpret_cast<float*>(spectrum.data_handle());
  const std::ptrdiff_t n_rows = static_cast<std::ptrdiff_t>(batch_) * dims_.input_nrow;
  const std::ptrdiff_t ncol = dims_.input_ncol;
#pragma omp parallel for
  for (std::ptrdiff_t r = 0; r < n_rows; r++) std::copy(in + r * ncol, in + (r + 1) * ncol, copy + r * ncol);
}

void convolution_ctx::convolve_spectrum(core::span1d<const complex_type> spectrum, float sigma,
                                        core::span3d<float> out) const
{
  FD_PROFILE_FN();
  assert(spectrum.size() == static_cast<std::size_t>(batch_) * dims_.freq_total());
  assert(out.is_exhaustive() && out.extent(0) == batch_);
  assert(out.extent(1) == dims_.input_nrow && out.extent(2) == dims_.input_ncol);

  const auto nrow = static_cast<std::size_t>(dims_.input_nrow);
  const auto ncol = static_cast<std::size_t>(dims_.input_ncol);
  const auto batch = static_cast<std::size_t>(batch_);
  // P - n >= reach: a wrapped tail crosses the zero strip, in either direction, before it can reach the image.
  const int reach = gaussian_reach(sigma);
  const auto row_len = static_cast<std::size_t>(next_fast_size(dims_.input_ncol + reach));  // padded row length
  const auto col_len = static_cast<std::size_t>(next_fast_size(dims_.input_nrow + reach));  // padded column length

  const auto kx = gaussian_kernel(ctx_, static_cast<int>(row_len), sigma);
  const auto ky = gaussian_kernel(ctx_, static_cast<int>(col_len), sigma);
  const ducc0::cmav<float, 1> kernel_x(kx.data_handle(), {row_len});
  const ducc0::cmav<float, 1> kernel_y(ky.data_handle(), {col_len});

  const auto* in = reinterpret_cast<const float*>(spectrum.data_handle());
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
        const std::size_t r0 = static_cast<std::size_t>(blk) * kTileLines;
        const std::size_t rb = std::min(kTileLines, n_rows - r0);
        for (std::size_t r = 0; r < rb; r++) {
          float* line = buf + r * row_len;
          std::copy(in + (r0 + r) * ncol, in + (r0 + r + 1) * ncol, line);
          std::fill(line + ncol, line + row_len, 0.0f);
        }
        const ducc0::vfmav<float> lines(buf, {rb, row_len});
        ducc0::convolve_axis(lines, lines, std::size_t{1}, kernel_x, 1);
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
        float* plane = dst + (static_cast<std::size_t>(blk) / blocks_per_plane) * nrow * ncol;
        const std::size_t c0 = (static_cast<std::size_t>(blk) % blocks_per_plane) * kTileLines;
        const std::size_t cb = std::min(kTileLines, ncol - c0);
        for (std::size_t c = 0; c < cb; c++) std::fill(buf + c * col_len + nrow, buf + (c + 1) * col_len, 0.0f);
        for (std::size_t i = 0; i < nrow; i++) {
          const float* src = plane + i * ncol + c0;
          for (std::size_t c = 0; c < cb; c++) buf[c * col_len + i] = src[c];
        }
        const ducc0::vfmav<float> lines(buf, {cb, col_len});
        ducc0::convolve_axis(lines, lines, std::size_t{1}, kernel_y, 1);
        for (std::size_t i = 0; i < nrow; i++) {
          float* row = plane + i * ncol + c0;
          for (std::size_t c = 0; c < cb; c++) row[c] = buf[c * col_len + i];
        }
      }
    }
  }
}

}  // namespace fast_deconv::linalg
