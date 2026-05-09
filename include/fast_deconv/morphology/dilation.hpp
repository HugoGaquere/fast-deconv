#pragma once

#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/morphology/roi.hpp>

namespace fast_deconv::morphology {
    
void binary_dilation(const core::stream_resources& stream_res, core::device_span2d<bool> data,
                     core::device_span2d<bool> structure, roi structure_roi, core::device_span2d<bool> out);
    
}