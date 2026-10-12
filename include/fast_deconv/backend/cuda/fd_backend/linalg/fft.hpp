#pragma once
#include <cufft.h>

#include <cstddef>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>

namespace fast_deconv::linalg {

using complex_type = cufftComplex;

// The padded and frequency buffers below are opaque scratch: flat, exhaustive,
// and sized from @p dims, so they are passed as raw pointers rather than spans.

// Pads a batch of images: each (input_nrow, input_ncol) image goes to the top-left of its
// (padded_nrow, padded_ncol) plane, the rest is zeroed.
void pad_async(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output, int n_batch);

// Crops a batch of (padded_nrow, padded_ncol) planes to their top-left (input_nrow, input_ncol).
void crop_async(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output, int n_batch);

/// out = input * Gaussian(sigma) / padded_total, over @p n_batch half-complex planes of @p dims.
void multiply_with_gaussian(const core::exec_ctx& ctx, const fft_dims& dims, int n_batch, const complex_type* input,
                            complex_type* out, float sigma);

}  // namespace fast_deconv::linalg
