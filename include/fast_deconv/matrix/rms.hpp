#pragma once

#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::matrix {

/**
 * @brief   Standard deviation of unmasked pixels of a 2D image (a.k.a. RMS in this project).
 * @details Computes sqrt(E[x^2] - E[x]^2) over the unmasked pixels. Pixels where
 *          @p mask is true are excluded. Mirrors Python `cp.std(d_unmasked)`.
 *          Synchronizes the stream before returning. Returns 0 if all pixels are masked.
 *
 * @param[in] resources   GPU memory allocator.
 * @param[in] stream_res  CUDA stream resources.
 * @param[in] data        Input image (must be exhaustive, same extents as @p mask).
 * @param[in] mask        Boolean mask (true = excluded).
 *
 * @return RMS / standard deviation value (on host).
 */
float rms(const core::resources& resources, const core::stream_resources& stream_res,
          core::device_span2d<float> data, core::device_span2d<bool> mask);

}  // namespace fast_deconv::matrix
