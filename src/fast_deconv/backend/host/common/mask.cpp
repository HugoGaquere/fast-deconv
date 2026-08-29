#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <emu/submdspan.hpp>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <fast_deconv/morphology/dilation.hpp>
#include <fast_deconv/morphology/roi.hpp>

namespace fast_deconv::common {

void mask_and_abs_async(const core::exec_ctx& ctx, core::span2d<float> data, core::span2d<bool> mask, float fill_value,
                        bool abs)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extents() == mask.extents());

  // Flat: indexing through operator() blocks if-conversion, hence vectorization.
  float* d = data.data_handle();
  const bool* m = mask.data_handle();
  for (std::size_t i = 0; i < data.size(); i++) d[i] = m[i] ? fill_value : (abs ? std::fabs(d[i]) : d[i]);
}

void mask_and_abs_async(const core::exec_ctx& ctx, core::span3d<float> data, core::span2d<bool> mask, float fill_value,
                        bool abs)
{
  assert(data.extent(1) == mask.extent(0) && data.extent(2) == mask.extent(1));

  for (int f = 0; f < data.extent(0); f++) {
    mask_and_abs_async(ctx, emu::submdspan(data, f), mask, fill_value, abs);
  }
}

void mask_and_abs_async(const core::exec_ctx& ctx, core::span3d<float> data, core::span3d<bool> mask, float fill_value,
                        bool abs)
{
  assert(data.extents() == mask.extents());

  for (int f = 0; f < data.extent(0); f++) {
    mask_and_abs_async(ctx, emu::submdspan(data, f), emu::submdspan(mask, f), fill_value, abs);
  }
}

void mask_less_than_threshold(const core::exec_ctx& ctx, core::span2d<float> data, float threshold, float fill_value)
{
  assert(data.is_exhaustive());

  float* d = data.data_handle();
  for (std::size_t i = 0; i < data.size(); i++) d[i] = d[i] < threshold ? fill_value : d[i];
}

void build_auto_mask(const core::exec_ctx& ctx, const std::vector<index2d>& coords, const std::vector<int>& scales,
                     core::span3d<float> central_facet_psfs, core::span1d<const float> weights_freq,
                     core::span1d<float> scale_sigmas, float fft_padding, core::span2d<bool> external_mask,
                     core::span3d<bool> mask_per_scale)
{
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

  // ---- 2. Build a PSF-sized convolve_ctx with batched plans over n_freq (1 R2C + 1 C2R) ----
  linalg::convolve_ctx conv(ctx, psf_nrow, psf_ncol, /*forward_batch=*/n_freq, /*backward_batch=*/n_freq,
                            /*n_backward_plans=*/1, fft_padding);

  core::owned_ptr<std::byte> fft_work_area;
  if (conv.required_work_size() > 0) fft_work_area = ctx.alloc_ptr_async<std::byte>(conv.required_work_size());
  conv.bind_work_area(fft_work_area.get());

  const linalg::fft_dims& dims = conv.dims();
  const int padded_total = dims.padded_total();
  const int freq_total = dims.freq_total();

  // ---- 3. Generate per-scale Gaussian kernels at PSF FFT size ----
  auto gauss_kernels = ctx.alloc_mdcontainer_async<float>(n_scales, dims.freq_nrow, dims.freq_ncol);
  scale::make_gaussian_kernels_async(ctx, scale_sigmas, dims.padded_ncol, gauss_kernels);

  // ---- 4. Batched pad+ifftshift and R2C of all frequency PSFs (once, reused across scales) ----
  auto padded_psf = ctx.alloc_ptr_async<float>(static_cast<std::size_t>(n_freq) * padded_total);
  auto freq_psf = ctx.alloc_ptr_async<linalg::complex_type>(static_cast<std::size_t>(n_freq) * freq_total);

  linalg::pad_ifftshift_batched_async(ctx, dims, central_facet_psfs.data_handle(), padded_psf.get(), n_freq);
  conv.forward_async(padded_psf.get(), freq_psf.get());

  // ---- 5. Per-scale: multiply -> C2R -> crop -> weighted mean -> FWHM -> dilate ----
  auto freq_conv2 = ctx.alloc_ptr_async<linalg::complex_type>(static_cast<std::size_t>(n_freq) * freq_total);
  auto padded_conv2 = ctx.alloc_ptr_async<float>(static_cast<std::size_t>(n_freq) * padded_total);
  auto conv2_cropped = ctx.alloc_mdcontainer_async<float>(n_freq, psf_nrow, psf_ncol);
  auto conv2_psf = ctx.alloc_mdcontainer_async<float>(psf_nrow, psf_ncol);
  auto fwhm_mask = ctx.alloc_mdcontainer_async<bool>(psf_nrow, psf_ncol);
  auto dilation_out = ctx.alloc_mdcontainer_async<bool>(dirty_nrow, dirty_ncol);

  const float norm = 1.0f / static_cast<float>(padded_total);

  for (int i = 0; i < n_scales; i++) {
    // 5a. Multiply by G^2 in freq domain (with norm); the kernel is 2D, broadcast across channels.
    const float* gauss = gauss_kernels.data_handle() + static_cast<std::int64_t>(i) * freq_total;
    for (int b = 0; b < n_freq; b++) {
      // Widened before the multiply: n_freq * freq_total can exceed INT_MAX.
      const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(b) * freq_total;
      for (int k = 0; k < freq_total; k++) {
        const float g2_norm = gauss[k] * gauss[k] * norm;
        const linalg::complex_type val = freq_psf[off + k];
        freq_conv2[off + k] = {val.real() * g2_norm, val.imag() * g2_norm};
      }
    }

    // 5b. Batched C2R back to space
    conv.backward_async(freq_conv2.get(), padded_conv2.get());

    // 5c. Batched fftshift + crop to PSF size
    linalg::fftshift_crop_async(ctx, dims, padded_conv2.get(), conv2_cropped.data_handle(), n_freq);

    // 5d. Weighted mean across channels -> 2D conv2_psf
    linalg::weighted_sum_async(ctx, conv2_cropped, weights_freq, conv2_psf);

    // 5e. FWHM threshold: bool out = (conv2_psf > max / 2)
    const float* psf_begin = conv2_psf.data_handle();
    const float threshold = *std::max_element(psf_begin, psf_begin + psf_npix) * 0.5f;
    for (int k = 0; k < psf_npix; k++) fwhm_mask.data_handle()[k] = psf_begin[k] > threshold;

    // 5f. Bounding box of FWHM, used as the structuring element's extent
    const roi structure_roi = morphology::compute_mask_roi(ctx, fwhm_mask);

    // 5g. Dilate mask_per_scale[i] using FWHM as structuring element
    core::span2d<bool> current_mask = emu::submdspan(mask_per_scale, i);
    morphology::binary_dilation(ctx, current_mask, fwhm_mask, structure_roi, dilation_out);

    // 5h. Copy dilation result back into mask_per_scale[i]
    std::memcpy(current_mask.data_handle(), dilation_out.data_handle(), dirty_npix * sizeof(bool));
  }

  // ---- 6. Negate to "true=masked" convention and OR external_mask into every scale slice ----
  for (int i = 0; i < n_scales; i++) {
    bool* slice = mask_per_scale.data_handle() + static_cast<std::int64_t>(i) * dirty_npix;
    for (int k = 0; k < dirty_npix; k++) slice[k] = (!slice[k]) || external_mask.data_handle()[k];
  }
}

}  // namespace fast_deconv::common
