#pragma once

#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::matrix {

/**
 * @brief   Max reduction over a masked 2D image, optionally over absolute values.
 * @details Pixels where @p mask is true are excluded from the reduction. When
 *          @p use_abs is true, the absolute value of each unmasked pixel is used.
 *          Synchronizes the stream before returning.
 *
 * @param[in] resources   GPU memory allocator.
 * @param[in] stream_res  CUDA stream resources.
 * @param[in] data        Input image (must be exhaustive, same extents as @p mask).
 * @param[in] mask        Boolean mask (true = excluded).
 * @param[in] use_abs     If true, take fabsf of each unmasked value before max.
 *
 * @return Max value (on host). Undefined if every pixel is masked.
 */
float max(const core::resources& resources, const core::stream_resources& stream_res,
          core::device_span2d<float> data, core::device_span2d<bool> mask, bool use_abs = false);

}  // namespace fast_deconv::matrix
