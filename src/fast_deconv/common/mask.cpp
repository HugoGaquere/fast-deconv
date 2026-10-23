#include <cassert>
#include <cstddef>
#include <cstdint>
#include <emu/submdspan.hpp>
#include <fast_deconv/common/detail/mask_steps.hpp>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/gaussian_convolution.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/morphology/dilation.hpp>
#include <fast_deconv/morphology/roi.hpp>
#include <numbers>

namespace fast_deconv::common {

void build_auto_mask(const core::exec_ctx& ctx, const std::vector<index2d>& coords, const std::vector<int>& scales,
                     core::span3d<const float> central_facet_psfs, core::span1d<const float> weights_freq,
                     const std::vector<float>& scale_sigmas, core::span2d<const bool> external_mask,
                     core::span3d<bool> mask_per_scale)
{
  FD_PROFILE_FN();
  assert(mask_per_scale.is_exhaustive());
  assert(external_mask.is_exhaustive());
  assert(central_facet_psfs.is_exhaustive());
  assert(coords.size() == scales.size());
  assert(weights_freq.extent(0) == central_facet_psfs.extent(0));
  assert(external_mask.extent(0) == mask_per_scale.extent(1) && external_mask.extent(1) == mask_per_scale.extent(2));

  const std::int64_t n_scales = mask_per_scale.extent(0);
  const std::int64_t dirty_nrow = mask_per_scale.extent(1);
  const std::int64_t dirty_ncol = mask_per_scale.extent(2);
  const std::int64_t psf_nrow = central_facet_psfs.extent(1);
  const std::int64_t psf_ncol = central_facet_psfs.extent(2);

  // ---- 1. Zero the output and stamp peaks ----
  detail::stamp_peaks(ctx, coords, scales, mask_per_scale);

  // ---- 2. Weighted mean of the channels' PSFs, transformed once and reused by every scale ----
  // Convolution is linear: sum_f w_f (G * psf_f) = G * (sum_f w_f psf_f), so one plane instead of n_freq.
  auto psf_mean = ctx.alloc_mdcontainer<float>(psf_nrow, psf_ncol);
  linalg::weighted_sum(ctx, central_facet_psfs, weights_freq, psf_mean);
  // Wrap-free for every scale's sigma * sqrt(2).
  const linalg::gaussian_convolution_ctx conv(ctx, /*batch=*/1, psf_nrow, psf_ncol, scale_sigmas,
                                              std::numbers::sqrt2_v<float>);
  auto spectrum = conv.make_spectrum();
  conv.forward(core::span3d<const float>(psf_mean.data_handle(), 1, psf_nrow, psf_ncol), spectrum);

  // ---- 3. Per-scale: conv2 -> FWHM -> dilate ----
  auto conv2_psf = ctx.alloc_mdcontainer<float>(psf_nrow, psf_ncol);
  auto fwhm_mask = ctx.alloc_mdcontainer<bool>(psf_nrow, psf_ncol);
  auto dilation_out = ctx.alloc_mdcontainer<bool>(dirty_nrow, dirty_ncol);

  for (std::int64_t i = 0; i < n_scales; i++) {
    // 3a. psf_mean ** G_s ** G_s, one convolution with G(sigma * sqrt(2)) since G(sigma)^2 = G(sigma * sqrt(2)).
    conv.convolve(spectrum, scale_sigmas.at(core::narrow<std::size_t>(i)) * std::numbers::sqrt2_v<float>,
                  core::span3d<float>(conv2_psf.data_handle(), 1, psf_nrow, psf_ncol));

    // 3b. FWHM threshold: bool out = (conv2_psf > max / 2)
    const float psf_max = matrix::find_peak(ctx, conv2_psf, {}).value;
    detail::threshold_above(ctx, conv2_psf, psf_max * 0.5f, fwhm_mask);

    // 3c. Bounding box of FWHM, used as the structuring element's extent
    const roi structure_roi = morphology::compute_mask_roi(ctx, fwhm_mask);

    // 3d. Dilate mask_per_scale[i] using FWHM as structuring element
    core::span2d<bool> current_mask = emu::submdspan(mask_per_scale, i);
    morphology::binary_dilation(ctx, current_mask, fwhm_mask, structure_roi, dilation_out);

    // 3e. Mask everything outside the dilated neighborhoods, plus the external mask
    detail::mask_outside(ctx, dilation_out, external_mask, current_mask);
  }
}

}  // namespace fast_deconv::common
