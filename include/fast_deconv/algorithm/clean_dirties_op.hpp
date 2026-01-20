#pragma once

#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/algorithm/detail/clean_dirties_op.cuh>

namespace fast_deconv::algo::wscms {

/*
 * Fused kernel that performs both dirty and scaled_dirty subtraction:
 * 1. dirty[i] -= psf[i] * coeffs[ch] * gain
 * 2. scaled_dirty[i] -= psf_2[i] * gain_scaled * mask[i]
 *
 * @param psf         Convolved PSF for dirty subtraction (nch, npol, h, w)
 * @param psf_2       Convolved PSF for scaled_dirty subtraction (nch, npol, h, w)
 * @param dirty       Dirty image to subtract from (in-place) (nch, npol, h, w)
 * @param scaled_dirty Scaled dirty image to subtract from (in-place) (nch, npol, h, w)
 * @param coeffs      Per-channel coefficients (nch,)
 * @param mask        Mask for scaled_dirty subtraction (nch, npol, h, w) - use 0.0/1.0 values
 * @param gain        Gain for dirty subtraction
 * @param gain_scaled Gain for scaled_dirty subtraction (typically peak_value * gain)
 */
void clean_dirties_async(
    core::device_span4d_fs& psf,
    core::device_span4d_fs& psf_2,
    core::device_span4d_fs& dirty,
    core::device_span4d_fs& scaled_dirty,
    core::device_vect_f& coeffs,
    core::device_span4d_fs& mask,
    float gain,
    core::stream_resources& resources)
{
  detail::clean_dirties_async(psf, psf_2, dirty, scaled_dirty, coeffs, mask, gain, resources);
}

}  // namespace fast_deconv::algo::wscms
