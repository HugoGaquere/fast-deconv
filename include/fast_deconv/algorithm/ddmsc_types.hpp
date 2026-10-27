#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/common/convergence.hpp>
#include <fast_deconv/common/region.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fast_deconv::algorithm::ddmsc {

struct params {
  // outer loop params
  int max_iteration = 0;           // total minor iterations across all scale selections
  float divergence_factor = 0.0f;  // flux growth ratio that counts as divergence

  // Every "rms" in this file is the standard deviation of the mean residual over all pixels (see
  // matrix::stats_result::rms), not its root mean square.

  // stopping criterion: stop_flux = max(flux_threshold, rms_factor*rms, peak_factor*peak, sidelobe_coeff*peak)
  // computed once per call from the initial peak/standard deviation of the mean residual
  float flux_threshold = 0.0f;       // absolute floor below which to stop
  float stop_rms_factor = 0.0f;      // weights the initial standard deviation in the stop-flux composition
  float stop_peak_factor = 0.0f;     // weights initial peak in the stop-flux composition
  float stop_cycle_factor = 0.0f;    // 0 disables the sidelobe contribution
  float stop_sidelobe_level = 0.0f;  // PSF sidelobe level used in the sidelobe term

  // clean loop params
  bool clean_negative = false;
  float peak_factor = 0.0f;     // sub-clean threshold = peak_value * peak_factor
  float gamma = 0.0f;           // CLEAN loop gain
  int max_clean_iteration = 0;  // sub-minor loop iterations per scale selection

  // scales params
  // Relative standard deviation change between outer iterations below which the selected scale takes a stall strike;
  // a strike-free iteration resets its count. More than 5 consecutive strikes retire the scale. Engaging the
  // auto-mask clears every count.
  float scale_stall_threshold = 0.0f;
  bool enable_auto_mask = false;        // master switch for auto-masking
  bool force_enable_auto_mask = false;  // with enable_auto_mask, engage masking immediately, bypassing thresholds

  // Auto-masking engages once the residual peak drops to the threshold. The peak threshold wins when both are
  // set; with neither, the threshold is 0, so only a non-positive peak (or force_enable_auto_mask) engages it.
  std::optional<float> auto_mask_peak_threshold;  // absolute flux
  std::optional<float> auto_mask_rms_threshold;   // multiple of the current standard deviation
};

struct context {
  core::exec_resources resources;
  core::exec_ctx compute_stream;  // scale search, PSF builds and the mean-residual clean loop
  core::exec_ctx aux_stream;      // clean-loop fit/subtract, overlapping compute_stream
  core::cont4d<const float> raw_psfs;
  core::cont2d<const float> xdes;
  core::cont2d<const bool> mask;
  std::vector<float> scale_sigmas;
  core::host_span1d<const float> scale_bias;
  core::host_span2d<const int> map_pixel_facet;
  std::int64_t dirty_nrow;
  std::int64_t dirty_ncol;

  std::vector<common::index2d> historical_peak_coords;  // across runs; feeds the auto-mask
  std::vector<int> historical_scales;                   // scale of each historical component

  /**
   * @brief Stages the run-constant inputs and copies the scale sigmas.
   *
   * The host spans must outlive the context: the host backend borrows them.
   *
   * @param[in] exec_device      Device the resources and both streams are created on.
   * @param[in] raw_psfs         Raw PSFs, host, (n_facets, n_freq, psf_h, psf_w). Staged.
   * @param[in] xdes             Spectral design matrix, host, (n_freq, n_order). Staged.
   * @param[in] mask             External mask, host, (dirty_nrow, dirty_ncol), true = masked. Staged.
   * @param[in] scale_sigmas     Gaussian sigma per scale, host, (n_scales,). Copied.
   * @param[in] scale_bias       Per-scale weight on the peak value during scale selection, host, (n_scales,). Borrowed.
   * @param[in] map_pixel_facet  Facet index of every dirty pixel, host, (dirty_nrow, dirty_ncol). Borrowed.
   * @param[in] dirty_nrow       Dirty image rows.
   * @param[in] dirty_ncol       Dirty image columns.
   */
  context(int exec_device, core::host_span4d<const float> raw_psfs, core::host_span2d<const float> xdes,
          core::host_span2d<const bool> mask, core::host_span1d<const float> scale_sigmas,
          core::host_span1d<const float> scale_bias, core::host_span2d<const int> map_pixel_facet,
          std::int64_t dirty_nrow, std::int64_t dirty_ncol)
      : resources(exec_device),
        compute_stream(resources.make_ctx()),
        aux_stream(resources.make_ctx()),
        raw_psfs(compute_stream.stage(raw_psfs)),
        xdes(compute_stream.stage(xdes)),
        mask(compute_stream.stage(mask)),
        scale_sigmas(scale_sigmas.data_handle(), scale_sigmas.data_handle() + scale_sigmas.size()),
        scale_bias(scale_bias),
        map_pixel_facet(map_pixel_facet),
        dirty_nrow(dirty_nrow),
        dirty_ncol(dirty_ncol)
  {
    compute_stream.wait();  // staging copies complete before construction returns
  }

  context(const context&) = delete;
  context& operator=(const context&) = delete;
  context(context&&) = delete;
  context& operator=(context&&) = delete;
};

struct ddmsc_result {
  std::vector<std::pair<std::int64_t, std::int64_t>>
      peak_coords;  // pair, not index2d: bound read-only to Python as list[tuple]
  std::vector<int> scales;
  std::vector<float> gains;
  std::vector<float> coeffs;  // row-major [component][n_order]
  std::int64_t n_order = 0;
  float initial_flux = 0.0f;  // peak flux of the mean residual on entry
  float initial_rms = 0.0f;   // standard deviation of the mean residual on entry
  // final_flux and final_rms ignore the auto-mask.
  float final_flux = 0.0f;   // peak flux of the mean residual after the last outer iteration
  float final_rms = 0.0f;    // standard deviation of the mean residual after the last outer iteration
  float stop_flux = 0.0f;    // composed stop-flux threshold used for this call (max of the four limits)
  int total_iterations = 0;  // total minor iterations consumed across all outer cycles
  int outer_iterations = 0;  // scale selections, including the one that found no component
  common::convergence_status status = common::convergence_status::running;  // why the outer loop ended

  /// @p capacity is the most components one call can produce.
  explicit ddmsc_result(std::size_t capacity)
  {
    peak_coords.reserve(capacity);
    scales.reserve(capacity);
    gains.reserve(capacity);
  }

  [[nodiscard]] std::size_t n_components() const { return peak_coords.size(); }

  /// Appends one component; its coefficients come later, in set_coeffs().
  void add_component(common::index2d coords, int scale, float gain)
  {
    peak_coords.emplace_back(coords.row, coords.col);
    scales.push_back(scale);
    gains.push_back(gain);
  }

  /// One row per component; @p ctx must be the stream @p rows was written on.
  void set_coeffs(const core::exec_ctx& ctx, core::span2d<const float> rows)
  {
    if (rows.extent(0) != std::ssize(peak_coords))
      throw std::logic_error("ddmsc_result: coefficient rows do not match the components added");

    n_order = rows.extent(1);
    coeffs.resize(rows.size());
    if (coeffs.empty()) return;
    ctx.copy(coeffs.data(), rows.data_handle(), std::ssize(coeffs));
    ctx.wait();
  }
};

}  // namespace fast_deconv::algorithm::ddmsc
