#pragma once

#include <cstdint>
#include <fast_deconv/algorithm/ddmsc_cycles.hpp>
#include <fast_deconv/algorithm/ddmsc_types.hpp>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>

namespace fast_deconv::algorithm::ddmsc {

/**
 * @brief   Stateful DDMSC deconvolution session.
 * @details Constructor creates the resource pool and both streams, and captures the
 *          run-constant `context` (raw_psfs, xdes, mask, scale_sigmas, scale_bias,
 *          map_pixel_facet). The scale and PSF convolution plans are built per `run()`.
 *
 *          `run()` accepts only the per-call inputs that change between major
 *          cycles: the dirty image, jones normalization, and per-frequency
 *          weights. All algorithm parameters are tunable.
 *
 *          All array inputs are host views. raw_psfs, xdes and mask are staged at
 *          construction, the `run()` inputs per call; scale_sigmas is copied; scale_bias
 *          and map_pixel_facet stay borrowed on the host. `dirty` is in/out — its residual
 *          is copied back into the caller's host buffer when `run()` returns.
 */
class Ddmsc {
 public:
  /**
   * @brief Builds the context (see context::context) and sets every algorithm parameter to its default.
   *
   * Staging borrows on the host backend, so there the staged inputs must outlive this too.
   *
   * @param[in] raw_psfs         Raw PSFs, host, (n_facets, n_freq, psf_h, psf_w). Staged.
   * @param[in] xdes             Spectral design matrix, host, (n_freq, n_order). Staged.
   * @param[in] mask             External mask, host, (dirty_nrow, dirty_ncol), true = masked. Staged.
   * @param[in] scale_sigmas     Gaussian sigma per scale, host, (n_scales,). Copied.
   * @param[in] scale_bias       Per-scale weight on the peak value during scale selection, host, (n_scales,).
   *                             Borrowed: must outlive this.
   * @param[in] map_pixel_facet  Facet index of every dirty pixel, host, (dirty_nrow, dirty_ncol). Borrowed: must
   *                             outlive this.
   * @param[in] dirty_nrow       Dirty image rows.
   * @param[in] dirty_ncol       Dirty image columns.
   * @param[in] exec_device      Device the resources and streams are created on.
   */
  Ddmsc(core::host_span4d<const float> raw_psfs, core::host_span2d<const float> xdes,
        core::host_span2d<const bool> mask, core::host_span1d<const float> scale_sigmas,
        core::host_span1d<const float> scale_bias, core::host_span2d<const int> map_pixel_facet,
        std::int64_t dirty_nrow, std::int64_t dirty_ncol, int exec_device = 0);

  /**
   * @brief Run the minor loop for one major cycle.
   *
   * @param[in,out] dirty         Dirty image, host, (n_freq, nrow, ncol); the residual is written back
   *                              before returning.
   * @param[in]     jones_norm    Jones normalization, host, (n_freq, nrow, ncol).
   * @param[in]     weights_freq  Per-frequency weights, host, (n_freq,).
   * @return Extracted components, entry/exit flux and standard deviation, and why the loop stopped.
   */
  ddmsc_result run(core::host_span3d<float> dirty, core::host_span3d<const float> jones_norm,
                   core::host_span1d<const float> weights_freq);

  // Getters and setters of the algorithm parameters, applied from the next run(). Each maps to the params field of
  // the same name (max_sub_iteration -> max_clean_iteration, stall_threshold -> scale_stall_threshold,
  // auto_mask -> enable_auto_mask, force_auto_mask -> force_enable_auto_mask); see params for their meaning.

  bool clean_negative() const { return params_.clean_negative; }
  void set_clean_negative(bool v) { params_.clean_negative = v; }

  float peak_factor() const { return params_.peak_factor; }
  void set_peak_factor(float v) { params_.peak_factor = v; }

  float gamma() const { return params_.gamma; }
  void set_gamma(float v) { params_.gamma = v; }

  int max_sub_iteration() const { return params_.max_clean_iteration; }
  void set_max_sub_iteration(int v) { params_.max_clean_iteration = v; }

  float flux_threshold() const { return params_.flux_threshold; }
  void set_flux_threshold(float v) { params_.flux_threshold = v; }

  float stop_rms_factor() const { return params_.stop_rms_factor; }
  void set_stop_rms_factor(float v) { params_.stop_rms_factor = v; }

  float stop_peak_factor() const { return params_.stop_peak_factor; }
  void set_stop_peak_factor(float v) { params_.stop_peak_factor = v; }

  float stop_cycle_factor() const { return params_.stop_cycle_factor; }
  void set_stop_cycle_factor(float v) { params_.stop_cycle_factor = v; }

  float stop_sidelobe_level() const { return params_.stop_sidelobe_level; }
  void set_stop_sidelobe_level(float v) { params_.stop_sidelobe_level = v; }

  int max_iteration() const { return params_.max_iteration; }
  void set_max_iteration(int v) { params_.max_iteration = v; }

  float divergence_factor() const { return params_.divergence_factor; }
  void set_divergence_factor(float v) { params_.divergence_factor = v; }

  float stall_threshold() const { return params_.scale_stall_threshold; }
  void set_stall_threshold(float v) { params_.scale_stall_threshold = v; }

  bool auto_mask() const { return params_.enable_auto_mask; }
  void set_auto_mask(bool v) { params_.enable_auto_mask = v; }

  bool force_auto_mask() const { return params_.force_enable_auto_mask; }
  void set_force_auto_mask(bool v) { params_.force_enable_auto_mask = v; }

  std::optional<float> auto_mask_peak_threshold() const { return params_.auto_mask_peak_threshold; }
  void set_auto_mask_peak_threshold(std::optional<float> v) { params_.auto_mask_peak_threshold = v; }

  std::optional<float> auto_mask_rms_threshold() const { return params_.auto_mask_rms_threshold; }
  void set_auto_mask_rms_threshold(std::optional<float> v) { params_.auto_mask_rms_threshold = v; }

 private:
  context ctx_;
  params params_;
};

}  // namespace fast_deconv::algorithm::ddmsc
