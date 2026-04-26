#pragma once

#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::common {

void mask_and_abs_async(const core::stream_resources& stream_res, core::device_span2d<float> data,
                        core::device_span2d<bool> mask, float fill_value, bool abs);

void mask_and_abs_async(const core::stream_resources& stream_res, core::device_span3d<float> data,
                        core::device_span2d<bool> mask, float fill_value, bool abs);

void mask_less_than_threshold(const core::stream_resources& stream_res, core::device_span2d<float> data,
                              float threshold, float fill_value);

}  // namespace fast_deconv::common
