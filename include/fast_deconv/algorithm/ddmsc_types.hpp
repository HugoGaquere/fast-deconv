#pragma once

#include <algorithm>
#include <cstddef>
#include <fast_deconv/algorithm/psf_convolution.hpp>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/common/convergence.hpp>
#include <fast_deconv/common/region.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fast_deconv::algorithm::ddmsc {

static constexpr int MAX_SPECTRAL_ORDER = 4;

enum class auto_mask_threshold_type {
  peak_value,
  rms,
};

struct params {
  // outer loop params
  int max_iteration;        // total minor iterations across all scale selections
  float divergence_factor;  // flux growth ratio that counts as divergence

  // stopping criterion: stop_flux = max(flux_threshold, rms_factor*rms, peak_factor*peak, sidelobe_coeff*peak)
  // computed once per call from the initial peak/RMS of the mean residual
  float flux_threshold;       // absolute floor below which to stop
  float stop_rms_factor;      // weights initial RMS in the stop-flux composition
  float stop_peak_factor;     // weights initial peak in the stop-flux composition
  float stop_cycle_factor;    // 0 disables the sidelobe contribution
  float stop_sidelobe_level;  // PSF sidelobe level used in the sidelobe term

  // clean loop params
  bool clean_negative;
  float peak_factor;        // sub-clean threshold = peak_value * peak_factor
  float gamma;              // CLEAN loop gain
  int max_clean_iteration;  // sub-minor loop iterations per scale selection

  // scales params
  float scale_stall_threshold;                    // RMS change below this counts as a stall
  bool enable_auto_mask;                          // master switch for auto-masking
  bool force_enable_auto_mask;                    // engage masking unconditionally, bypassing thresholds
  std::optional<float> auto_mask_peak_threshold;  // engage when residual peak <= this (absolute flux)
  std::optional<float> auto_mask_rms_threshold;   // engage when residual peak <= this * running RMS

  // conv-PSF cache params
  psf_cache_mode psf_cache_policy;  // when the convolved PSFs get built
};

struct context {
  core::exec_resources resources;
  core::exec_ctx compute_stream;  // scale search, PSF builds and the mean-residual clean loop
  core::exec_ctx aux_stream;      // clean-loop fit/subtract, overlapping compute_stream
  core::cont4d<float> raw_psfs;
  core::cont2d<float> xdes;
  core::cont2d<bool> mask;
  std::vector<float> scale_sigmas;
  core::host_span1d<float> scale_bias;
  core::host_span2d<int> map_pixel_facet;
  int dirty_nrow;
  int dirty_ncol;
  int n_freq;
  float fft_padding;

  std::vector<common::index2d> historical_peak_coords;  // across runs; feeds the auto-mask
  std::vector<int> historical_scales;                   // scale of each historical component

  /// Validates the dimensions, uploads the run-constant inputs and copies the scale sigmas.
  context(int exec_device, const core::host_span4d<float>& raw_psfs, const core::host_span2d<float>& xdes,
          const core::host_span2d<bool>& mask, const core::host_span1d<float>& scale_sigmas,
          const core::host_span1d<float>& scale_bias, const core::host_span2d<int>& map_pixel_facet, int dirty_nrow,
          int dirty_ncol, int n_freq, float fft_padding)
      : resources(checked_device(exec_device, raw_psfs, dirty_nrow, dirty_ncol)),
        compute_stream(resources.make_ctx()),
        aux_stream(resources.make_ctx()),
        raw_psfs(compute_stream.copy_of(raw_psfs)),
        xdes(compute_stream.copy_of(xdes)),
        mask(compute_stream.copy_of(mask)),
        scale_sigmas(scale_sigmas.data_handle(), scale_sigmas.data_handle() + scale_sigmas.size()),
        scale_bias(scale_bias),
        map_pixel_facet(map_pixel_facet),
        dirty_nrow(dirty_nrow),
        dirty_ncol(dirty_ncol),
        n_freq(n_freq),
        fft_padding(fft_padding)
  {
    compute_stream.wait();  // staging copies, before the host sources go
  }

  context(const context&) = delete;
  context& operator=(const context&) = delete;
  context(context&&) = delete;
  context& operator=(context&&) = delete;

 private:
  // Runs before resources: an oversized plane throws before any device allocation.
  static int checked_device(int exec_device, const core::host_span4d<float>& raw_psfs, int dirty_nrow, int dirty_ncol)
  {
    core::check_plane_fits_int32(dirty_nrow, dirty_ncol, "dirty");
    core::check_plane_fits_int32(raw_psfs.extent(2), raw_psfs.extent(3), "psf");
    return exec_device;
  }
};

struct ddmsc_result {
  std::vector<std::pair<int, int>> peak_coords;  // pair, not index2d: bound read-only to Python as list[tuple]
  std::vector<int> scales;
  std::vector<float> gains;
  std::vector<std::vector<float>> coeffs;
  float final_flux = 0.0f;   // peak flux of the mean residual after the last outer iteration
  float stop_flux = 0.0f;    // composed stop-flux threshold used for this call (max of the four limits)
  int total_iterations = 0;  // total minor iterations consumed across all outer cycles
  common::convergence_status status = common::convergence_status::running;  // why the outer loop ended

  /// @p capacity is the most components one call can produce.
  explicit ddmsc_result(std::size_t capacity)
  {
    peak_coords.reserve(capacity);
    scales.reserve(capacity);
    gains.reserve(capacity);
    coeffs.reserve(capacity);
  }

  void add_component(common::index2d coords, int scale, float gain)
  {
    peak_coords.emplace_back(coords.row, coords.col);
    scales.push_back(scale);
    gains.push_back(gain);
  }

  void add_coeffs_from_device(const core::exec_ctx& ctx, core::span2d<const float> rows)
  {
    const std::size_t n_components = rows.extent(0);
    const std::size_t n_order = rows.extent(1);
    if (n_components != peak_coords.size() - coeffs.size())
      throw std::logic_error("ddmsc_result: coefficient rows do not match the components added");

    std::vector<float> staged(n_components * n_order);
    ctx.copy(staged.data(), rows.data_handle(), staged.size());
    ctx.wait();

    for (std::size_t i = 0; i < n_components; ++i) {
      coeffs.emplace_back(staged.begin() + i * n_order, staged.begin() + (i + 1) * n_order);
    }
  }
};

}  // namespace fast_deconv::algorithm::ddmsc
