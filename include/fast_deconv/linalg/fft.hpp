#pragma once
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>

// Resolved by the include path: CMake puts backend/${FAST_DECONV_BACKEND}
// on it, and every backend provides this file.
#include <fd_backend/linalg/fft.hpp>

namespace fast_deconv::linalg {

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

}  // namespace fast_deconv::linalg
