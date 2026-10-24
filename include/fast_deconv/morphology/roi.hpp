#pragma once

#include <fast_deconv/common/region.hpp>
#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::morphology {

/**
 * @brief Bounding box of the set pixels of @p data. Syncs @p ctx.
 *
 * @param[in] ctx   Execution context.
 * @param[in] data  Mask, device, (nrow, ncol).
 * @return Half-open box [rmin, rmax) x [cmin, cmax); all zeros (empty) if no pixel is set.
 */
common::roi compute_mask_roi(const core::exec_ctx& ctx, core::span2d<const bool> data);

}  // namespace fast_deconv::morphology
