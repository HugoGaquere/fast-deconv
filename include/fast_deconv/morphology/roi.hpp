#pragma once

#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::morphology {

/// Bounding box of a mask's foreground, half-open: rows [rmin, rmax), columns [cmin, cmax).
struct roi {
  std::int64_t rmin, rmax, cmin, cmax;
};

/**
 * @brief Bounding box of the set pixels of @p data. Syncs @p ctx.
 *
 * @param[in] ctx   Execution context.
 * @param[in] data  Mask, device, (nrow, ncol).
 * @return Half-open box [rmin, rmax) x [cmin, cmax); all zeros (empty) if no pixel is set.
 */
roi compute_mask_roi(const core::exec_ctx& ctx, core::span2d<const bool> data);

}  // namespace fast_deconv::morphology
