#pragma once

#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/matrix/peak.hpp>

namespace fast_deconv::matrix {

/// Argmax of criterion(data[i], i) over the plane; value is the criterion at the peak. Syncs @p ctx.
peak find_peak(const core::exec_ctx& ctx, core::span2d<const float> data, peak_criterion criterion);

}  // namespace fast_deconv::matrix
