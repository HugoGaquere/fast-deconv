#pragma once
#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::linalg {

void weighted_sum_async(const core::exec_ctx& ctx, const float* A, const float* weights, float* out, int w, int n);

void weighted_sum_async(const core::exec_ctx& ctx, const core::span3d<float> A, const core::span1d<float> weights,
                        core::span2d<float> out);

}  // namespace fast_deconv::linalg