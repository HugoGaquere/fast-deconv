#pragma once
#include <cufft.h>

#include <cstddef>
#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>

namespace fast_deconv::linalg {

using complex_type = cufftComplex;

// The padded and frequency buffers below are opaque scratch: flat, exhaustive,
// and sized from @p dims, so they are passed as raw pointers rather than spans.

/**
 * @brief Pads a batch of images to the top-left of zeroed planes.
 *
 * @param[in]  ctx      Context the kernel runs on.
 * @param[in]  dims     Input and padded sizes.
 * @param[in]  input    n_batch images of (input_nrow, input_ncol).
 * @param[out] output   n_batch planes of (padded_nrow, padded_ncol), written entirely.
 * @param[in]  n_batch  Number of images.
 */
void pad(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output, std::int64_t n_batch);

/// Crops a batch of (padded_nrow, padded_ncol) planes to their top-left (input_nrow, input_ncol).
void crop(const core::exec_ctx& ctx, const fft_dims& dims, const float* input, float* output, std::int64_t n_batch);

/// out = input * Gaussian(sigma) / padded_total, over @p n_batch half-complex planes of @p dims.
void multiply_with_gaussian(const core::exec_ctx& ctx, const fft_dims& dims, std::int64_t n_batch,
                            const complex_type* input, complex_type* out, float sigma);

}  // namespace fast_deconv::linalg
