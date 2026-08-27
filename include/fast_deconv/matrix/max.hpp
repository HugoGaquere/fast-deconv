#pragma once

#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::matrix {

float max(const core::exec_ctx& ctx, core::span2d<float> data, core::span2d<bool> mask, bool use_abs = false);

}  // namespace fast_deconv::matrix
