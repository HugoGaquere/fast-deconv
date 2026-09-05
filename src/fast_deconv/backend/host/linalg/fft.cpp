#include <omp.h>
#include <pocketfft_hdronly.h>

#include <algorithm>
#include <cstddef>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/fft.hpp>

namespace pf = pocketfft;

namespace fast_deconv::linalg {

namespace {
// pocketfft strides are in bytes, one entry per axis of a (n_batch, nrow, ncol) row-major array.
template <typename T>
pf::stride_t byte_strides(int nrow, int ncol)
{
  const std::ptrdiff_t item = sizeof(T);
  return {nrow * ncol * item, ncol * item, item};
}

// Axis 0 is the batch and stays untransformed; the r2c/c2r axis must come last.
const pf::shape_t kFftAxes{1, 2};

// One knob for both pools: pocketfft caps this by the transform's own parallelism.
std::size_t fft_threads() { return static_cast<std::size_t>(omp_get_max_threads()); }
}  // namespace

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

// fct = 1 keeps cuFFT's unnormalized convention: callers apply 1/padded_total themselves.
void convolve_ctx::forward_async(float* input, complex_type* output) const
{
  FD_PROFILE_FN();
  const pf::shape_t shape{static_cast<std::size_t>(forward_batch_), static_cast<std::size_t>(dims_.padded_nrow),
                          static_cast<std::size_t>(dims_.padded_ncol)};
  pf::r2c(shape, byte_strides<float>(dims_.padded_nrow, dims_.padded_ncol),
          byte_strides<complex_type>(dims_.freq_nrow, dims_.freq_ncol), kFftAxes, pf::FORWARD, input, output, 1.0f,
          fft_threads());
}

// plan_idx selects one of the cuFFT plans; pocketfft caches its own, so it is inert here.
void convolve_ctx::backward_async(complex_type* input, float* output, int plan_idx) const
{
  FD_PROFILE_FN();
  const pf::shape_t shape{static_cast<std::size_t>(backward_batch_), static_cast<std::size_t>(dims_.padded_nrow),
                          static_cast<std::size_t>(dims_.padded_ncol)};
  pf::c2r(shape, byte_strides<complex_type>(dims_.freq_nrow, dims_.freq_ncol),
          byte_strides<float>(dims_.padded_nrow, dims_.padded_ncol), kFftAxes, pf::BACKWARD, input, output, 1.0f,
          fft_threads());
}

}  // namespace fast_deconv::linalg
