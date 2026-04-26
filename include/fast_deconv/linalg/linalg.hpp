#pragma once
#include <fast_deconv/core/resources.hpp>

#include "fast_deconv/core/span_types.hpp"

namespace fast_deconv::linalg {

void weighted_sum_async(const core::stream_resources& stream_res, const float* A,
                        const float* weights, float* out, int w, int n);

void weighted_sum_async(const core::stream_resources& stream_res,
                        const core::device_span3d<float> A, const core::device_vect<float> weights,
                        core::device_span2d<float> out);

}  // namespace fast_deconv::linalg