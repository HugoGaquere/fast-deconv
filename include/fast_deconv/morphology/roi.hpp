#pragma once

#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::morphology {

struct roi {
  int xmin, xmax, ymin, ymax;
};

roi compute_mask_roi(const core::exec_ctx& ctx, core::span2d<bool> data);

}  // namespace fast_deconv::morphology