#pragma once

#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/morphology/roi.hpp>

namespace fast_deconv::morphology {

void binary_dilation(const core::exec_ctx& ctx, core::span2d<bool> data, core::span2d<bool> structure,
                     roi structure_roi, core::span2d<bool> out);

}