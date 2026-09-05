#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <emu/submdspan.hpp>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <limits>
#include <stdexcept>

namespace fast_deconv::scale {

namespace {
constexpr float kPiSquared = 9.869604403f;
}  // namespace

void make_gaussian_kernels_async(const core::exec_ctx& ctx, core::span1d<float> sigmas, int scale_ncol_full,
                                 core::span3d<float> scales)
{
  const int n_scales = scales.extent(0);
  const int scale_nrow = scales.extent(1);
  const int scale_ncol_half = scales.extent(2);

  // Collapsed: the PSF path builds one kernel at a time, so n_scales alone can be 1.
#pragma omp parallel for collapse(2)
  for (int s = 0; s < n_scales; s++) {
    for (int i = 0; i < scale_nrow; i++) {
      const float sigma_squarred = sigmas(s) * sigmas(s);
      const float freq_row = i < (scale_nrow + 1) / 2 ? static_cast<float>(i) / scale_nrow
                                                      : static_cast<float>(i - scale_nrow) / scale_nrow;
      const float freq_row_squarred = freq_row * freq_row;
      for (int j = 0; j < scale_ncol_half; j++) {
        const float freq_col = static_cast<float>(j) / scale_ncol_full;
        const float rhosq = freq_row_squarred + freq_col * freq_col;
        scales(s, i, j) = std::exp(-2.0f * kPiSquared * rhosq * sigma_squarred);
      }
    }
  }
}

void convolve_with_scales(const linalg::convolve_ctx& conv, core::span2d<float> dirty, core::span3d<float> scales,
                          core::span3d<float> out_scaled_dirty)
{
  const core::exec_ctx& ctx = conv.ctx();  // plans run on this lane
  const linalg::fft_dims& dims = conv.dims();
  const int n_scales = out_scaled_dirty.extent(0);
  const int img_padded_total = dims.padded_total();
  const int freq_total = dims.freq_total();
  const int npix = dims.input_total();

  // Scale 0 (sigma == 0) is the identity kernel — copy the input directly.
  std::memcpy(out_scaled_dirty.data_handle(), dirty.data_handle(), npix * sizeof(float));

  if (n_scales <= 1) return;

  // Plan was built with this batch size; loop over chunks of that size to cover
  // all (n_scales - 1) non-trivial scales. (n_scales - 1) must be a multiple of
  // backward_batch — the cuFFT C2R plan always processes exactly batch slices,
  // so a remainder would OOB-read scales[] and OOB-write out_scaled_dirty[].
  const int chunk_batch = conv.backward_batch();
  const int total_scales = n_scales - 1;
  if (total_scales % chunk_batch != 0) {
    throw std::invalid_argument("convolve_with_scales: (n_scales - 1) must be a multiple of backward_batch_size");
  }

  // Plans were bound to their stream at construction (the ddmsc context's
  // compute stream); temporaries are stream-ordered on that same stream.
  auto dirty_padded = ctx.alloc_ptr_async<float>(img_padded_total);
  auto dirty_freq = ctx.alloc_ptr_async<linalg::complex_type>(freq_total);
  auto scaled_dirty_freq =
      ctx.alloc_ptr_async<linalg::complex_type>(static_cast<std::size_t>(chunk_batch) * freq_total);
  auto scaled_dirty = ctx.alloc_ptr_async<float>(static_cast<std::size_t>(chunk_batch) * img_padded_total);

  // Pad + ifftshift dirty image
  linalg::pad_ifftshift_async(ctx, dims, dirty.data_handle(), dirty_padded.get());

  // Forward R2C — once, reused across all chunks
  conv.forward_async(dirty_padded.get(), dirty_freq.get());

  // Loop over chunks of chunk_batch scales (skip scale 0, identity, already memcpy'd above).
  const float norm = 1.0f / static_cast<float>(img_padded_total);
  for (int chunk_off = 0; chunk_off < total_scales; chunk_off += chunk_batch) {
    const int scale_idx_start = 1 + chunk_off;

    // Source is absolute (scale index), destination is chunk-relative.
    const float* chunk_scales = scales.data_handle() + static_cast<std::int64_t>(scale_idx_start) * freq_total;
    // Collapsed: one team for the whole chunk, not one fork per slice.
#pragma omp parallel for collapse(2)
    for (int b = 0; b < chunk_batch; b++) {
      for (int i = 0; i < freq_total; i++) {
        // Widened before the multiply: n_batch * freq_total can exceed INT_MAX.
        const std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(b) * freq_total + i;
        const linalg::complex_type dirty_val = dirty_freq[i];
        const float scale_norm = chunk_scales[idx] * norm;
        scaled_dirty_freq[idx] = {dirty_val.real() * scale_norm, dirty_val.imag() * scale_norm};
      }
    }

    conv.backward_async(scaled_dirty_freq.get(), scaled_dirty.get());

    linalg::fftshift_crop_async(ctx, dims, scaled_dirty.get(),
                                out_scaled_dirty.data_handle() + static_cast<std::int64_t>(scale_idx_start) * npix,
                                chunk_batch);
  }
}

int scale_selection(const core::exec_ctx& ctx, core::span3d<float> scaled_dirty, core::host_span1d<float> bias,
                    const std::vector<int>& retired_scales)
{
  assert(scaled_dirty.is_exhaustive());

  const int n_scales = scaled_dirty.extent(0);
  const int npix = scaled_dirty.extent(1) * scaled_dirty.extent(2);

  // Masked-out pixels are already -inf, so they never win the per-scale max.
  int best_scale = 0;
  float best_biased = -std::numeric_limits<float>::infinity();
  for (int s = 0; s < n_scales; s++) {
    if (std::find(retired_scales.begin(), retired_scales.end(), s) != retired_scales.end()) continue;

    const float* plane = scaled_dirty.data_handle() + static_cast<std::int64_t>(s) * npix;
    // max is exact and associative, so the reduction matches max_element bit for bit.
    float plane_max = -std::numeric_limits<float>::infinity();
#pragma omp parallel for reduction(max : plane_max)
    for (int i = 0; i < npix; i++) plane_max = std::max(plane_max, plane[i]);

    const float biased = plane_max * bias(s);
    if (biased > best_biased) {
      best_biased = biased;
      best_scale = s;
    }
  }

  return best_scale;
}

void convolve_psf_with_scale_async(const linalg::convolve_ctx& conv, core::span3d<float> psf,
                                   core::span1d<float> d_sigma, int scale_idx, core::span1d<const float> weights,
                                   psf_convolve_scratch& scratch, core::span3d<float> out_conv_psf,
                                   core::span2d<float> out_conv2_mean)
{
  const core::exec_ctx& ctx = conv.ctx();  // plans run on this lane
  const linalg::fft_dims& dims = conv.dims();
  const int n_freq = psf.extent(0);
  const int psf_npix = dims.input_total();
  const int padded_total = dims.padded_total();
  const int freq_total = dims.freq_total();

  // Scale 0 fast path: no convolution needed
  if (scale_idx == 0) {
    std::memcpy(out_conv_psf.data_handle(), psf.data_handle(),
                sizeof(float) * static_cast<std::size_t>(n_freq) * psf_npix);
    linalg::weighted_sum_async(ctx, core::span3d<const float>(psf), weights, out_conv2_mean);
    return;
  }

  // Gaussian scale kernel at PSF resolution, memoized across facets of one scale
  if (scratch.kernel_scale_idx != scale_idx) {
    make_gaussian_kernels_async(ctx, d_sigma, dims.padded_ncol, scratch.scale_kernel);
    scratch.kernel_scale_idx = scale_idx;
  }

  const float norm = 1.0f / static_cast<float>(padded_total);

  // 1. Pad + ifftshift (batched over n_freq)
  linalg::pad_ifftshift_batched_async(ctx, dims, psf.data_handle(), scratch.padded_psf.get(), n_freq);

  // 2. Batched R2C FFT
  conv.forward_async(scratch.padded_psf.get(), scratch.freq_psf.get());

  // 3. Multiply by G and G^2 — the kernel is 2D, broadcast across channels.
  // Collapsed: n_freq alone is a handful of channels, too few to fill the pool.
#pragma omp parallel for collapse(2)
  for (int b = 0; b < n_freq; b++) {
    for (int i = 0; i < freq_total; i++) {
      // Widened before the multiply: n_freq * freq_total can exceed INT_MAX.
      const std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(b) * freq_total + i;
      const float g_norm = scratch.scale_kernel.data_handle()[i] * norm;
      const float g2_norm = scratch.scale_kernel.data_handle()[i] * g_norm;
      const linalg::complex_type val = scratch.freq_psf[idx];
      scratch.freq_conv[idx] = {val.real() * g_norm, val.imag() * g_norm};
      scratch.freq_conv2[idx] = {val.real() * g2_norm, val.imag() * g2_norm};
    }
  }

  // 4. Batched C2R IFFT for conv_psf
  conv.backward_async(scratch.freq_conv.get(), scratch.padded_conv.get());

  // 5. Batched C2R IFFT for conv2_psf
  conv.backward_async(scratch.freq_conv2.get(), scratch.padded_conv2.get(), /*plan_idx=*/1);

  // 6. fftshift + crop for conv_psf -> output
  linalg::fftshift_crop_async(ctx, dims, scratch.padded_conv.get(), out_conv_psf.data_handle(), n_freq);

  // 7. fftshift + crop for conv2_psf -> temporary
  linalg::fftshift_crop_async(ctx, dims, scratch.padded_conv2.get(), scratch.conv2_cropped.data_handle(), n_freq);

  // 8. Weighted mean over channels -> conv2_mean output
  linalg::weighted_sum_async(ctx, scratch.conv2_cropped, weights, out_conv2_mean);
}

}  // namespace fast_deconv::scale
