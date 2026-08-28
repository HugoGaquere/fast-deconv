#pragma once

#include <fast_deconv/common/region.hpp>
#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::morphology {

common::roi compute_mask_roi(const core::exec_ctx& ctx, core::span2d<bool> data);

}  // namespace fast_deconv::morphology
