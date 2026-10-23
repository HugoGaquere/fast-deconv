#pragma once

#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/morphology/roi.hpp>

namespace fast_deconv::morphology {

/**
 * @brief Binary dilation of @p data by the part of @p structure inside @p structure_roi.
 *
 * The structuring element is the roi's (h, w) box, anchored at its center (h / 2, w / 2): out(r, c) is set where
 * data(r, c) is set, or where data(r - h / 2 + i, c - w / 2 + j) && structure(roi.rmin + i, roi.cmin + j) for some
 * (i, j) in the box. Pixels outside the image count as unset.
 *
 * @param[in]  ctx            Execution context; the work is enqueued on it.
 * @param[in]  data           Image to dilate, device, (nrow, ncol).
 * @param[in]  structure      Structuring-element plane, device.
 * @param[in]  structure_roi  Box of @p structure to use, half-open (see compute_mask_roi()).
 * @param[out] out            Dilated image, device, (nrow, ncol). Must be all false on entry: the CUDA backend
 *                            only sets pixels (the host backend clears it first). Must not alias @p data.
 */
void binary_dilation(const core::exec_ctx& ctx, core::span2d<const bool> data, core::span2d<const bool> structure,
                     common::roi structure_roi, core::span2d<bool> out);

}  // namespace fast_deconv::morphology