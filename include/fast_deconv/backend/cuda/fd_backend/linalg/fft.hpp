#pragma once
#include <cufft.h>

#include <cstddef>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>

namespace fast_deconv::linalg {

using complex_type = cufftComplex;

// The padded and frequency buffers below are opaque scratch: flat, exhaustive,
// and sized from @p dims, so they are passed as raw pointers rather than spans.

// Pads and ifftshifts a 2D image in one pass.
// Input:  (input_nrow, input_ncol) real, origin at center
// Output: (padded_nrow, padded_ncol) real, origin at (0,0), zero-padded
// Output is zeroed before launch.
void pad_ifftshift_async(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output);

// Pads and ifftshifts a batched 2D image in one pass.
// Input:  (n_batch, input_nrow, input_ncol) real, origin at center
// Output: (n_batch, padded_nrow, padded_ncol) real, origin at (0,0), zero-padded
// Output is zeroed before launch.
void pad_ifftshift_batched_async(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output,
                                 int n_batch);

// Fftshifts and crops a batched 2D image in one pass.
// Input:  (n_batch, padded_nrow, padded_ncol) real, origin at (0,0) (FFT output)
// Output: (n_batch, input_nrow, input_ncol) real, origin at center, cropped
void fftshift_crop_async(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output,
                         int n_batch);

/// out = input * Gaussian(sigma) / padded_total, over @p n_batch half-complex planes of @p dims.
void multiply_with_gaussian(const core::exec_ctx& ctx, const fft_dims& dims, int n_batch, const complex_type* input,
                            complex_type* out, float sigma);

}  // namespace fast_deconv::linalg
