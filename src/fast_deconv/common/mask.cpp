#include <cassert>
#include <cstddef>
#include <cstdint>
#include <emu/submdspan.hpp>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/core/macro.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/gaussian_convolution.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/morphology/dilation.hpp>
#include <fast_deconv/morphology/roi.hpp>
#include <fd_backend/core/for_each.hpp>
#include <iterator>
#include <numbers>
#include <vector>

namespace fast_deconv::common {

void build_auto_mask(const core::exec_ctx& ctx, const std::vector<core::index2d>& coords, const std::vector<int>& scales,
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
  // Flat offsets on the host, staged: the scatter then needs neither the coordinates nor the strides.
  ctx.zero(mask_per_scale);
  std::vector<std::int64_t> offsets;
  offsets.reserve(coords.size());
  for (std::size_t i = 0; i < coords.size(); i++)
    offsets.push_back(mask_per_scale.mapping()(scales.at(i), coords.at(i).row, coords.at(i).col));
  if (!offsets.empty()) {
    const std::int64_t n_coords = std::ssize(offsets);
    const auto d_offsets = ctx.stage(core::host_span1d<const std::int64_t>(offsets.data(), n_coords));
    const std::int64_t* off = d_offsets.data_handle();
    bool* stamped = mask_per_scale.data_handle();
    core::for_each_index(ctx, n_coords, [=] FD_HOST_DEVICE(std::int64_t i) { stamped[off[i]] = true; });
  }

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
    const float half_max = matrix::find_peak(ctx, conv2_psf, {}).value * 0.5f;
    const float* psf = conv2_psf.data_handle();
    bool* fwhm = fwhm_mask.data_handle();
    core::for_each_index(ctx, psf_nrow * psf_ncol, [=] FD_HOST_DEVICE(std::int64_t k) { fwhm[k] = psf[k] > half_max; });

    // 3c. Bounding box of FWHM, used as the structuring element's extent
    const core::roi structure_roi = morphology::compute_mask_roi(ctx, fwhm_mask);

    // 3d. Dilate mask_per_scale[i] using FWHM as structuring element
    core::span2d<bool> current_mask = emu::submdspan(mask_per_scale, i);
    morphology::binary_dilation(ctx, current_mask, fwhm_mask, structure_roi, dilation_out);

    // 3e. Mask everything outside the dilated neighborhoods, plus the external mask
    const bool* dilated = dilation_out.data_handle();
    const bool* external = external_mask.data_handle();
    bool* out = current_mask.data_handle();
    core::for_each_index(ctx, dirty_nrow * dirty_ncol,
                         [=] FD_HOST_DEVICE(std::int64_t k) { out[k] = !dilated[k] || external[k]; });
  }
}

void combine_scale_masks(const core::exec_ctx& ctx, core::span3d<const bool> mask_per_scale,
                         const std::vector<int>& scales, core::span2d<bool> out)
{
  FD_PROFILE_FN();
  assert(mask_per_scale.is_exhaustive() && out.is_exhaustive());
  assert(out.extent(0) == mask_per_scale.extent(1) && out.extent(1) == mask_per_scale.extent(2));
  assert(mask_per_scale.extent(0) <= 64);

  // The selected scales as a bitmask: bit s set = scale s selected.
  // 64 bits cap n_scales at 64 (asserted above).
  //
  // Building it:
  // std::uint64_t{1} << s is a number whose only set bit is bit s.
  // OR-ing (|=) it into selected turns that bit on.
  //
  // Example for scales = {0,2,3}
  // start                selected = 0b0000
  // s=0: 1<<0 = 0b0001 → selected = 0b0001
  // s=2: 1<<2 = 0b0100 → selected = 0b0101
  // s=3: 1<<3 = 0b1000 → selected = 0b1101
  //
  // Reading it in the kernel:
  // if (selected >> s & 1)
  // selected >> s shifts bit s down to position 0,
  // & 1 keeps only that bit. The result is 1 when scale s is selected.
  //
  // With selected = 0b1101:
  // s=0: 0b1101 & 1 = 1 → used
  // s=1: 0b0110 & 1 = 0 → skipped
  // s=2: 0b0011 & 1 = 1 → used
  // s=3: 0b0001 & 1 = 1 → used

  std::uint64_t selected = 0;
  for (const int s : scales) selected |= std::uint64_t{1} << s;

  const std::int64_t scale_stride = mask_per_scale.stride(0);
  const std::int64_t n_scales = mask_per_scale.extent(0);
  const bool* masks = mask_per_scale.data_handle();
  const std::int64_t n_elements = core::count(out);
  bool* combined = out.data_handle();
  core::for_each_index(ctx, n_elements, [=] FD_HOST_DEVICE(std::int64_t i) {
    bool masked = true;
    for (std::int64_t s = 0; s < n_scales; s++)
      if (selected >> s & 1) masked = masked && masks[s * scale_stride + i];
    combined[i] = masked;
  });
}

}  // namespace fast_deconv::common
