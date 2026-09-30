#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <emu/submdspan.hpp>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/convolution.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <fast_deconv/morphology/dilation.hpp>
#include <fast_deconv/morphology/roi.hpp>
#include <numbers>

namespace fast_deconv::common {

void build_auto_mask(const core::exec_ctx& ctx, const std::vector<index2d>& coords, const std::vector<int>& scales,
                     core::span3d<float> central_facet_psfs, core::span1d<const float> weights_freq,
                     const std::vector<float>& scale_sigmas, float fft_padding, core::span2d<bool> external_mask,
                     core::span3d<bool> mask_per_scale)
{
  FD_PROFILE_FN();
  assert(mask_per_scale.is_exhaustive());
  assert(external_mask.is_exhaustive());
  assert(central_facet_psfs.is_exhaustive());
  assert(coords.size() == scales.size());
  assert(weights_freq.extent(0) == central_facet_psfs.extent(0));
  assert(external_mask.extent(0) == mask_per_scale.extent(1) && external_mask.extent(1) == mask_per_scale.extent(2));

  const int n_scales = mask_per_scale.extent(0);
  const int dirty_nrow = mask_per_scale.extent(1);
  const int dirty_ncol = mask_per_scale.extent(2);
  const int dirty_npix = dirty_nrow * dirty_ncol;
  const int n_freq = central_facet_psfs.extent(0);
  const int psf_nrow = central_facet_psfs.extent(1);
  const int psf_ncol = central_facet_psfs.extent(2);
  const int psf_npix = psf_nrow * psf_ncol;

  // ---- 1. Zero the output and stamp peaks ----
  std::fill_n(mask_per_scale.data_handle(), mask_per_scale.size(), false);

  for (std::size_t i = 0; i < coords.size(); i++) {
    mask_per_scale(scales.at(i), coords.at(i).row, coords.at(i).col) = true;
  }

  // ---- 2. One batched forward transform of every channel's PSF, reused by every scale ----
  const linalg::convolution_ctx conv(ctx, psf_nrow, psf_ncol, fft_padding, /*batch=*/n_freq);
  auto spectrum =
      ctx.alloc_mdcontainer_async<linalg::complex_type>(static_cast<std::size_t>(n_freq) * conv.dims().freq_total());
  conv.forward(central_facet_psfs, spectrum);

  // ---- 3. Per-scale: conv2 -> weighted mean -> FWHM -> dilate ----
  auto conv2_cropped = ctx.alloc_mdcontainer_async<float>(n_freq, psf_nrow, psf_ncol);
  auto conv2_psf = ctx.alloc_mdcontainer_async<float>(psf_nrow, psf_ncol);
  auto fwhm_mask = ctx.alloc_mdcontainer_async<bool>(psf_nrow, psf_ncol);
  auto dilation_out = ctx.alloc_mdcontainer_async<bool>(dirty_nrow, dirty_ncol);

  for (int i = 0; i < n_scales; i++) {
    // 3a. psf ** G_s ** G_s, one convolution with G(sigma * sqrt(2)) since G(sigma)^2 = G(sigma * sqrt(2)).
    conv.convolve_spectrum(spectrum, scale_sigmas.at(i) * std::numbers::sqrt2_v<float>, conv2_cropped);

    // 3b. Weighted mean across channels -> 2D conv2_psf
    linalg::weighted_sum_async(ctx, conv2_cropped, weights_freq, conv2_psf);

    // 3c. FWHM threshold: bool out = (conv2_psf > max / 2)
    const float* psf_begin = conv2_psf.data_handle();
    const float threshold = *std::max_element(psf_begin, psf_begin + psf_npix) * 0.5f;
    for (int k = 0; k < psf_npix; k++) fwhm_mask.data_handle()[k] = psf_begin[k] > threshold;

    // 3d. Bounding box of FWHM, used as the structuring element's extent
    const roi structure_roi = morphology::compute_mask_roi(ctx, fwhm_mask);

    // 3e. Dilate mask_per_scale[i] using FWHM as structuring element
    core::span2d<bool> current_mask = emu::submdspan(mask_per_scale, i);
    morphology::binary_dilation(ctx, current_mask, fwhm_mask, structure_roi, dilation_out);

    // 3f. Write back negated to "true=masked", with external_mask OR'd in
    bool* slice = current_mask.data_handle();
    const bool* dilated = dilation_out.data_handle();
    const bool* external = external_mask.data_handle();
    for (int k = 0; k < dirty_npix; k++) slice[k] = !dilated[k] || external[k];
  }
}

}  // namespace fast_deconv::common
