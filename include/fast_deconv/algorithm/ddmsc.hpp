#pragma once

#include <fast_deconv/algorithm/ddmsc_cycles.hpp>
#include <fast_deconv/algorithm/ddmsc_types.hpp>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::algorithm::ddmsc {

/**
 * @brief   Stateful DDMSC deconvolution session.
 * @details Constructor builds heavy resources once: the GPU resource pool and
 *          the cuFFT plans for the scale and PSF convolutions. The static
 *          `DDMSC_ctx` (raw_psfs, xdes, scale_mask, scale_sigmas, scale_bias,
 *          map_pixel_facet) is also captured at construction.
 *
 *          `run()` accepts only the per-call inputs that change between major
 *          cycles: the dirty image, jones normalization, and per-frequency
 *          weights. All algorithm parameters are tunable.
 *
 *          All array inputs are host views: the constructor and `run()` copy
 *          them host->device internally. `dirty` is in/out — its residual is
 *          copied back into the caller's host buffer when `run()` returns.
 */
class Ddmsc {
 public:
  Ddmsc(const core::host_span4d<float>& raw_psfs, const core::host_span2d<float>& xdes,
        const core::host_span2d<bool>& mask, const core::host_vect<float>& scale_sigmas,
        const core::host_vect<float>& scale_bias, const core::host_span2d<int>& map_pixel_facet, int dirty_nrow,
        int dirty_ncol, int n_freq, float fft_padding, int exec_device = 0);

  /// Run one full deconvolution session. The updated residual is written back
  /// into @p dirty (host, in/out) before returning.
  ddmsc_result run(core::host_span3d<float>& dirty, const core::host_span3d<float>& jones_norm,
                   const core::host_vect<float>& weights_freq);

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
