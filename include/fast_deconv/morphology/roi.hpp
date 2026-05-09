#pragma once

#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::morphology {

struct roi {
  int xmin, xmax, ymin, ymax;
};

roi compute_mask_roi(const core::stream_resources& resources, core::device_span2d<bool> data);

}  // namespace fast_deconv::morphology