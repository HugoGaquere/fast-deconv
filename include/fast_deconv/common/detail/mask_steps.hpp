#pragma once

#include <fast_deconv/common/region.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <vector>

// The backend-specific steps of build_auto_mask(); every backend implements them.
namespace fast_deconv::common::detail {

/// Zero @p mask_per_scale, then set mask_per_scale(scales[i], coords[i].row, coords[i].col) for every i.
void stamp_peaks(const core::exec_ctx& ctx, const std::vector<index2d>& coords, const std::vector<int>& scales,
                 core::span3d<bool> mask_per_scale);

/// out = in > threshold, elementwise over two planes of the same shape.
void threshold_above(const core::exec_ctx& ctx, core::span2d<const float> in, float threshold, core::span2d<bool> out);

/// out = !region || external: masks every pixel outside @p region, plus the externally masked ones.
void mask_outside(const core::exec_ctx& ctx, core::span2d<const bool> region, core::span2d<const bool> external,
                  core::span2d<bool> out);

}  // namespace fast_deconv::common::detail
