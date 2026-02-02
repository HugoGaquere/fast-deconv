#pragma once

#include <fast_deconv/algorithm/detail/wscms_op.cuh>
#include <fast_deconv/core/dispatcher.hpp>
#include <fast_deconv/core/kernel_traits.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>

namespace fast_deconv::algo::wscms {

// psf shape   : nch, npol, height, width
// dirty shape : nch, npol, height, width

/*
 * Compute: out[k, 0, :, :] = psf[k, 0, :, :] * coeffs[k] * gain
 */
void subtract_psf_from_dirty_async(core::device_span4d_fs& psf,
                                   core::device_span4d_fs& dirty,
                                   core::device_vect_f& coeffs,
                                   core::device_span4d_fs& out,
                                   float gain,
                                   core::stream_resources& resources)
{
  // TODO: check psf.extent(0) == dirty.extent(0) == coeffs.size()
  core::dispatch<core::subtract_psf_from_dirty_tag>(
    resources, detail::subtract_psf_from_dirty_async, psf, dirty, coeffs, out, gain);
  // detail::subtract_psf_from_dirty_async(psf, dirty, coeffs, out, gain, resources);
}

// void subtract_psf_from_dirty(
//     core::device_span4d_fs& psf,
//     // core::device_span4d_fs& dirty,
//     // core::device_vect_f& coeffs,
//     // core::device_span4d_fs& out,
//     float gain,
//     core::stream_resources& resources)
// {
//   fmt::println("TEST");
//   // subtract_psf_from_dirty_async(psf, dirty, coeffs, out, gain, resources);
//   resources.sync();
// }

}  // namespace fast_deconv::algo::wscms
