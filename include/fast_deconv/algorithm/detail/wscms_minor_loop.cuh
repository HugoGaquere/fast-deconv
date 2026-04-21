#pragma once
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cub/cub.cuh>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fast_deconv/util/mdspan_utils.hpp>

#include "clean_minor_loop.cuh"
#include "scale.cuh"

namespace fast_deconv::algorithm::wscms::detail {

/**
 * @brief   Run a single minor cycle: scale selection + on-the-fly PSF convolution
 *          + sub-minor iterations.
 * @details Convolves the mean residual with all scale kernels, selects the best
 *          scale via biased peak-finding, computes the scale-convolved PSFs
 *          on-the-fly, then runs sub-minor iterations.
 *
 * @param[in]     resources      GPU memory allocator.
 * @param[in,out] dirty          Multi-frequency dirty image (n_freq, n_stokes, nrow, ncol).
 * @param[in]     mean_residual  Mean residual image pointer, size nrow * ncol.
 * @param[in]     jones_norm     Jones normalization.
 * @param[in]     weights_freq   Per-frequency weights.
 * @param[in]     scale_kernels  Pre-computed Gaussian scale kernels in frequency domain.
 * @param[in,out] wscms_ctx      WSCMS context (raw_psfs, masks, biases, gains, etc.).
 * @param[in]     scale_ctx      Scale convolution context (FFT plans, padding).
 * @param[in]     psf_ctx        PSF convolution context (FFT plans for PSF-sized convolutions).
 * @param[in]     params         Algorithm parameters.
 * @param[out]    d_coeffs_out   Device buffer for spectral coefficients (caller-allocated).
 * @param[out]    metas_out      Host vector to append per-component metadata to.
 *
 * @return Number of components produced.
 */
int wscms_minor_cycle(const core::resources& resources, core::device_span4d<float>& dirty,
                      float* mean_residual, const core::device_span4d<float>& jones_norm,
                      const core::device_vect<float>& weights_freq, float* scale_kernels,
                      WSCMS_ctx& wscms_ctx, const scale_convole_ctx& scale_ctx,
                      const psf_convolve_ctx& psf_ctx, WSCMS_params params, float* d_coeffs_out,
                      std::vector<component_meta>& metas_out)
{
  bool per_scale_mask = false;  // TODO: FIX THAT
  const auto& stream_r = resources.get_stream_resources();
  const int n_scales = params.n_scales;
  const int dirty_nrows = dirty.extent(2);
  const int dirty_ncols = dirty.extent(3);
  const int npix = dirty_nrows * dirty_ncols;
  const int n_freq = dirty.extent(0);
  const int n_facets = wscms_ctx.raw_psfs.extent(0);
  const int nch = wscms_ctx.raw_psfs.extent(1);
  const int psf_nrow = psf_ctx.psf_nrow;
  const int psf_ncol = psf_ctx.psf_ncol;
  const int psf_npix = psf_nrow * psf_ncol;

  FD_LOG_DEBUG(
      "wscms_minor_cycle: dirty={}x{} n_freq={} n_facets={} nch={} psf={}x{} psf_npix={} "
      "n_scales={}",
      dirty_nrows, dirty_ncols, n_freq, n_facets, nch, psf_nrow, psf_ncol, psf_npix, n_scales);

  float* scaled_mean_dirty = resources.alloc_async<float>(npix, stream_r);

  // 1. Convolve dirty image with all scale kernels
  float* scales_x_dirty = resources.alloc_async<float>(npix * n_scales, stream_r);
  scale_convolve(resources, stream_r, scale_ctx, mean_residual, scale_kernels, scales_x_dirty,
                 n_scales);

  // 2. Select the best scale
  scale_selection_result sel =
      scale_selection(resources, stream_r, scales_x_dirty, wscms_ctx.scale_masks.data_handle(),
                      wscms_ctx.scale_bias.data_handle(), n_scales, dirty_nrows, dirty_ncols,
                      params.clean_negative, per_scale_mask, params.forbidden_scales);

  // 3. Copy the winning slice to output
  copy_scale_slice(stream_r, scales_x_dirty, scaled_mean_dirty, sel.best_scale, npix);

  // 4. Cleanup scale selection temporaries
  resources.free_async(scales_x_dirty, stream_r);

  FD_LOG_INFO("selected scale_idx={} peak={:.6f} at ({},{})", sel.best_scale, sel.best_peak,
              sel.best_row, sel.best_col);

  // 5. Compute convolved PSFs for the selected scale
  float* conv_psfs = resources.alloc_async<float>(n_facets * nch * psf_npix, stream_r);
  float* conv2_psfs = resources.alloc_async<float>(n_facets * psf_npix, stream_r);

  convolve_psfs_for_scale(resources, stream_r, psf_ctx, wscms_ctx.raw_psfs.data_handle(),
                          wscms_ctx.scale_sigmas.data_handle() + sel.best_scale, sel.best_scale,
                          weights_freq.data_handle(), n_facets, nch, conv_psfs, conv2_psfs);

  // 6. Compute per-facet gains from convolved PSFs
  std::vector<float> h_scale_gains =
      compute_scale_gains(resources, stream_r, conv_psfs, weights_freq.data_handle(), n_facets, nch,
                          psf_npix, sel.best_scale, params.gamma);

  FD_LOG_DEBUG("scale_gains (n_facets={}): [{}]", n_facets, fmt::join(h_scale_gains, ", "));

  // 7. Run sub-minor loop with single-scale PSFs
  int n_components =
      wscms_subminor_cycles(resources, dirty, scaled_mean_dirty, conv_psfs, conv2_psfs, n_facets,
                            psf_nrow, psf_ncol, jones_norm, weights_freq, sel.best_scale,
                            h_scale_gains.data(), wscms_ctx, params, d_coeffs_out, metas_out);

  // 8. Free temporaries
  resources.free_async(conv2_psfs, stream_r);
  resources.free_async(conv_psfs, stream_r);
  resources.free_async(scaled_mean_dirty, stream_r);

  return n_components;
}

/**
 * @brief   Run the full minor-cycle loop: repeated scale selection with stall/divergence checks.
 * @details Allocates mean_residual internally and computes it from the dirty image at each
 *          iteration. PSFs are convolved on-the-fly for each selected scale.
 *
 * @param[in]     resources    GPU memory allocator.
 * @param[in,out] dirty        Multi-frequency dirty image (n_freq, n_stokes, nrow, ncol).
 * @param[in]     jones_norm   Jones normalization.
 * @param[in]     weights_freq Per-frequency weights.
 * @param[in,out] wscms_ctx    WSCMS context (raw_psfs, masks, biases, gains, etc.).
 * @param[in]     scale_ctx    Scale convolution context (FFT plans, padding).
 * @param[in]     psf_ctx      PSF convolution context (FFT plans for PSF-sized convolutions).
 * @param[in]     mask         Boolean mask for peak/rms computation, device, size npix.
 * @param[in]     params       Algorithm parameters (including outer-loop params).
 *
 * @return wscms_result with all extracted components, exit reason, and iteration count.
 */
wscms_result wscms_minor_cycles(const core::resources& resources, core::device_span4d<float>& dirty,
                                const core::device_span4d<float>& jones_norm,
                                const core::device_vect<float>& weights_freq, WSCMS_ctx& wscms_ctx,
                                const scale_convole_ctx& scale_ctx, const psf_convolve_ctx& psf_ctx,
                                const bool* mask, WSCMS_params params)
{
  const auto& stream_r = resources.get_stream_resources();
  const uint32_t n_freq = dirty.extent(0);
  const uint32_t dirty_nrows = dirty.extent(2);
  const uint32_t dirty_ncols = dirty.extent(3);
  const uint32_t npix = dirty_nrows * dirty_ncols;
  const uint32_t freq_stride = dirty.extent(1) * npix;  // n_stokes * nrow * ncol
  const uint64_t freq_scales_total =
      static_cast<int64_t>(scale_ctx.freq_nrow) * scale_ctx.freq_ncol * params.n_scales;

  FD_LOG_DEBUG(
      "wscms_minor_cycles: dirty={}x{} n_freq={} freq_stride={} npix={} freq_scales_total={} "
      "scale_freq={}x{} img_padded_ncol={}",
      dirty_nrows, dirty_ncols, n_freq, freq_stride, npix, freq_scales_total, scale_ctx.freq_nrow,
      scale_ctx.freq_ncol, scale_ctx.img_padded_ncol);

  float* mean_residual = resources.alloc_async<float>(npix, stream_r);

  // Allocate scale kernels once (constant across iterations)
  FD_LOG_INFO("Make scales: freq_scales_total={}", freq_scales_total);
  float* scale_kernels = resources.alloc_async<float>(freq_scales_total, stream_r);
  make_scales(resources, stream_r, wscms_ctx.scale_sigmas.data_handle(), scale_ctx.freq_nrow,
              scale_ctx.freq_ncol, scale_ctx.img_padded_ncol, params.n_scales, scale_kernels);

  // Shared device buffer for all component coefficients across all outer iterations.
  // The outer loop enters when total_iterations < max_iteration, and each outer cycle
  // may append up to max_sub_iteration components, so the worst-case final write offset
  // is (max_iteration - 1 + max_sub_iteration) * n_order.
  const int n_order = wscms_ctx.xdes.extent(1);
  const std::size_t coeffs_capacity =
      static_cast<std::size_t>(params.max_iteration + params.max_sub_iteration) * n_order;
  float* d_all_coeffs = resources.alloc_async<float>(coeffs_capacity, stream_r);
  stream_r.sync();
  std::vector<component_meta> all_metas;
  all_metas.reserve(params.max_iteration + params.max_sub_iteration);

  FD_LOG_DEBUG("coeffs buffer: n_order={} coeffs_capacity={} ({} floats = {} bytes)", n_order,
               coeffs_capacity, coeffs_capacity, coeffs_capacity * sizeof(float));

  // Initial mean residual
  FD_LOG_INFO("Compute mean residual");
  compute_mean_residual(stream_r, mean_residual, dirty.data_handle(), weights_freq.data_handle(),
                        n_freq, freq_stride, npix);

  // Initial flux and RMS
  FD_LOG_INFO("Compute peak flux");
  float track_flux =
      compute_peak_flux(resources, stream_r, mean_residual, mask, npix, params.clean_negative);
  float track_rms = compute_rms(resources, stream_r, mean_residual, mask, npix);

  FD_LOG_INFO("wscms_minor_cycles: initial_flux={:.8f} initial_rms={:.8f} stop_flux={:.8f}",
              track_flux, track_rms, params.stop_flux);

  // Tracking state
  wscms_result result;
  int total_iterations = 0;
  int diverged_count = 0;
  std::vector<int> retired_scales(params.forbidden_scales);
  std::vector<int> scale_stall_count(params.n_scales, 0);
  std::vector<bool> scales_stalled(params.n_scales, false);
  for (int s : params.forbidden_scales) {
    if (s >= 0 && s < params.n_scales) scales_stalled[s] = true;
  }

  wscms_exit_reason exit_reason = wscms_exit_reason::max_iterations;

  while (total_iterations < params.max_iteration) {
    // Check flux threshold
    if (track_flux <= params.stop_flux) {
      exit_reason = wscms_exit_reason::flux_threshold;
      FD_LOG_INFO("wscms_minor_cycles: flux {:.8f} <= stop_flux {:.8f}, stopping", track_flux,
                  params.stop_flux);
      break;
    }

    // Run one minor cycle (scale selection + on-the-fly PSF convolution + sub-minor iterations)
    params.forbidden_scales = retired_scales;
    const std::size_t coeffs_offset = static_cast<std::size_t>(total_iterations) * n_order;
    float* d_coeffs_cursor = d_all_coeffs + coeffs_offset;

    FD_LOG_DEBUG(
        "outer iter start: total_iterations={} coeffs_offset={} retired_scales={} track_flux={:.8f}"
        " track_rms={:.8f}",
        total_iterations, coeffs_offset, retired_scales.size(), track_flux, track_rms);

    int n_subminor =
        wscms_minor_cycle(resources, dirty, mean_residual, jones_norm, weights_freq, scale_kernels,
                          wscms_ctx, scale_ctx, psf_ctx, params, d_coeffs_cursor, all_metas);

    if (n_subminor == 0) {
      exit_reason = wscms_exit_reason::stalled;
      FD_LOG_INFO("wscms_minor_cycles: no components found, stopping");
      break;
    }

    int i_scale = all_metas.back().scale_idx;
    total_iterations += n_subminor;

    // Recompute mean residual from dirty
    compute_mean_residual(stream_r, mean_residual, dirty.data_handle(), weights_freq.data_handle(),
                          n_freq, freq_stride, npix);

    float this_flux =
        compute_peak_flux(resources, stream_r, mean_residual, mask, npix, params.clean_negative);

    float this_rms = compute_rms(resources, stream_r, mean_residual, mask, npix);

    FD_LOG_INFO("wscms_minor_cycles: [iter={}] flux={:.8f} rms={:.8f} scale={}", total_iterations,
                this_flux, this_rms, i_scale);
    FD_LOG_DEBUG(
        "outer iter end: n_subminor={} i_scale={} diverged_count={} scale_stall_count[{}]={}",
        n_subminor, i_scale, diverged_count, i_scale, scale_stall_count[i_scale]);

    // Divergence check
    if (std::abs(this_flux) > params.divergence_factor * std::abs(track_flux)) {
      diverged_count++;
      if (diverged_count > 5) {
        exit_reason = wscms_exit_reason::diverged;
        FD_LOG_INFO("wscms_minor_cycles: diverged after {} counts", diverged_count);
        break;
      }
    } else {
      diverged_count = 0;
    }

    // Stall check (per-scale)
    if (std::abs(track_rms - this_rms) < params.stall_threshold) {
      scale_stall_count[i_scale]++;
      if (scale_stall_count[i_scale] > 5) {
        retired_scales.push_back(i_scale);
        scales_stalled[i_scale] = true;
        FD_LOG_INFO("wscms_minor_cycles: retired scale {} due to stall", i_scale);

        // Check if all scales are stalled
        bool all_stalled = true;
        for (int s = 0; s < params.n_scales; s++) {
          if (!scales_stalled[s]) {
            all_stalled = false;
            break;
          }
        }
        if (all_stalled) {
          exit_reason = wscms_exit_reason::stalled;
          FD_LOG_INFO("wscms_minor_cycles: all scales stalled, stopping");
          break;
        }
      }
    }

    track_flux = this_flux;
    track_rms = this_rms;
  }

  // Final bulk D2H copy of all coefficients and sky_component construction
  int total_components = static_cast<int>(all_metas.size());
  if (total_components > 0) {
    std::vector<float> h_all_coeffs(total_components * n_order);
    CHECK_CUDA(cudaMemcpyAsync(h_all_coeffs.data(), d_all_coeffs,
                               total_components * n_order * sizeof(float), cudaMemcpyDeviceToHost,
                               stream_r.cuda_stream));
    stream_r.sync();

    result.components.reserve(total_components);
    for (int i = 0; i < total_components; i++) {
      result.components.push_back(sky_component{
          .row = all_metas[i].row,
          .col = all_metas[i].col,
          .scale_idx = all_metas[i].scale_idx,
          .gain = all_metas[i].gain,
          .coeffs = std::vector<float>(h_all_coeffs.begin() + i * n_order,
                                       h_all_coeffs.begin() + (i + 1) * n_order),
      });
    }
  }

  // Cleanup
  resources.free_async(d_all_coeffs, stream_r);
  resources.free_async(scale_kernels, stream_r);
  resources.free_async(mean_residual, stream_r);
  stream_r.sync();

  result.final_flux = track_flux;
  result.total_iterations = total_iterations;
  result.exit_reason = exit_reason;
  return result;
}

}  // namespace fast_deconv::algorithm::wscms::detail
