#pragma once
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>
#include <fd_backend/linalg/fft.hpp>

namespace fast_deconv::linalg {

void weighted_sum_async(const core::exec_ctx& ctx, const float* A, const float* weights, float* out, int w, int n);

void weighted_sum_async(const core::exec_ctx& ctx, const core::span3d<const float> A,
                        const core::span1d<const float> weights, core::span2d<float> out);

/// out = input * Gaussian(sigma) / padded_total, over @p n_batch half-complex planes of @p dims.
void multiply_with_gaussian(const core::exec_ctx& ctx, const fft_dims& dims, int n_batch, const complex_type* input,
                            complex_type* out, float sigma);

/// Same as multiply_with_gaussian, also writing out_conv2 = input * Gaussian(sigma)^2 / padded_total.
void multiply_with_gaussian_once_and_twice(const core::exec_ctx& ctx, const fft_dims& dims, int n_batch,
                                           const complex_type* input, complex_type* out_conv, complex_type* out_conv2,
                                           float sigma);

}  // namespace fast_deconv::linalg
