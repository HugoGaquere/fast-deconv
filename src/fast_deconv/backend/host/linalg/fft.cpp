#include <ducc0/fft/fft.h>
#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/convolution.hpp>
#include <fast_deconv/linalg/fft.hpp>
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

}  // namespace fast_deconv::linalg
