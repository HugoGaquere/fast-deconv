#pragma once

#include <cufft.h>

#include <algorithm>
#include <cstddef>
#include <fast_deconv/common/convergence.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "fast_deconv/linalg/fft.hpp"

namespace fast_deconv::algorithm::ddmsc {

static constexpr int MAX_SPECTRAL_ORDER = 4;

/// Image-domain scale convolution context: one R2C plan (batch=1, forward FFT
/// of mean dirty) + one batched C2R plan (batch=backward_batch_size).
/// `convolve_with_scales` loops over chunks of `backward_batch_size` scales
/// per IFFT call — pick the batch size to balance throughput vs. memory.
struct scale_convolve_ctx : public linalg::convolve_ctx {
  /// Build plans for a single forward FFT and a batched backward FFT of
  /// @p backward_batch_size at a time. The plans run on @p stream_res;
  /// `convolve_with_scales` must drive the convolution on that same stream
  /// and ensure (n_scales - 1) is a multiple of @p backward_batch_size. The
  /// owner must bind_work_area() before the first convolution.
  scale_convolve_ctx(const core::stream_resources& stream_res, int nrow, int ncol, int backward_batch_size,
                     float padding)
      : linalg::convolve_ctx(stream_res, nrow, ncol, /*forward_batch=*/1,
                             /*backward_batch=*/std::max(1, backward_batch_size), /*n_backward_plans=*/1, padding)
  {
  }

  cufftHandle& plan_forward() { return plans_forward[0]; }
  cufftHandle plan_forward() const { return plans_forward[0]; }
  cufftHandle& plan_backward() { return plans_backward[0]; }
  cufftHandle plan_backward() const { return plans_backward[0]; }
};

/// PSF-domain convolution context: batched R2C + two C2R plans (one for conv, one for conv^2).
struct psf_convolve_ctx : public linalg::convolve_ctx {
  /// Build batched plans (over n_freq channels). The plans run on
  /// @p stream_res; the owner must bind_work_area() before the first use.
  psf_convolve_ctx(const core::stream_resources& stream_res, int psf_nrow, int psf_ncol, int nch, float padding)
      : linalg::convolve_ctx(stream_res, psf_nrow, psf_ncol, /*forward_batch=*/nch, /*backward_batch=*/nch,
                             /*n_backward_plans=*/2, padding)
  {
  }

  cufftHandle& plan_forward() { return plans_forward[0]; }
  cufftHandle plan_forward() const { return plans_forward[0]; }
  cufftHandle& plan_backward() { return plans_backward[0]; }
  cufftHandle plan_backward() const { return plans_backward[0]; }
  cufftHandle& plan_backward_2() { return plans_backward[1]; }
  cufftHandle plan_backward_2() const { return plans_backward[1]; }
};

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
};

struct device_state {
  core::resources exec_resources;
  core::stream_resources compute_stream;  // drives the convolution path; the FFT plans bind to it
  core::stream_resources aux_stream;      // clean-loop fit/subtract, overlapping compute_stream
  core::device_cont4d<float> raw_psfs_d;
  core::device_cont2d<float> xdes_d;
  core::device_cont2d<bool> mask_d;
  core::device_cont<float> scale_sigmas_d;
  core::device_ptr<std::byte> fft_work_area;  // shared by both plan sets, run sequentially
  scale_convolve_ctx scale_convolve;
  psf_convolve_ctx psf_convolve;

  device_state(int exec_device, const core::host_span4d<float>& raw_psfs, const core::host_span2d<float>& xdes,
               const core::host_span2d<bool>& mask, const core::host_vect<float>& scale_sigmas, int dirty_nrow,
               int dirty_ncol, int n_freq, float fft_padding)
      : exec_resources(exec_device),
        compute_stream(exec_resources.make_stream()),
        aux_stream(exec_resources.make_stream()),
        raw_psfs_d(compute_stream.copy_h2d_async(raw_psfs)),
        xdes_d(compute_stream.copy_h2d_async(xdes)),
        mask_d(compute_stream.copy_h2d_async(mask)),
        scale_sigmas_d(compute_stream.copy_h2d_async(scale_sigmas)),
        scale_convolve(compute_stream, dirty_nrow, dirty_ncol,
                       /*backward_batch_size=*/static_cast<int>(scale_sigmas.size()) - 1, fft_padding),
        psf_convolve(compute_stream, raw_psfs_d.extent(2), raw_psfs_d.extent(3), n_freq, fft_padding)
  {
    const size_t shared_work_size = std::max(scale_convolve.required_work_size(), psf_convolve.required_work_size());
    if (shared_work_size > 0) fft_work_area = compute_stream.alloc_ptr_async<std::byte>(shared_work_size);
    compute_stream.sync();  // staging copies and the work-area alloc, before the host sources go
    scale_convolve.bind_work_area(fft_work_area.get());
    psf_convolve.bind_work_area(fft_work_area.get());
  }

  device_state(const device_state&) = delete;
  device_state& operator=(const device_state&) = delete;
  device_state(device_state&&) = delete;
  device_state& operator=(device_state&&) = delete;
};

struct context {
  int exec_device;
  core::host_span4d<float> raw_psfs;
  core::host_span2d<float> xdes;
  core::host_span2d<bool> mask;
  core::host_vect<float> scale_sigmas;
  core::host_vect<float> scale_bias;
  core::host_span2d<int> map_pixel_facet;
  int dirty_nrow;
  int dirty_ncol;
  int n_freq;
  float fft_padding;

  std::vector<std::pair<int, int>> historical_peak_coords;  // across runs; feeds the auto-mask
  std::vector<int> historical_scales;                       // scale of each historical component

  /// Validates the dimensions and stores the inputs; allocates nothing.
  context(int exec_device, const core::host_span4d<float>& raw_psfs, const core::host_span2d<float>& xdes,
          const core::host_span2d<bool>& mask, const core::host_vect<float>& scale_sigmas,
          const core::host_vect<float>& scale_bias, const core::host_span2d<int>& map_pixel_facet, int dirty_nrow,
          int dirty_ncol, int n_freq, float fft_padding)
      : exec_device(exec_device),
        raw_psfs(raw_psfs),
        xdes(xdes),
        mask(mask),
        scale_sigmas(scale_sigmas),
        scale_bias(scale_bias),
        map_pixel_facet(map_pixel_facet),
        dirty_nrow(dirty_nrow),
        dirty_ncol(dirty_ncol),
        n_freq(n_freq),
        fft_padding(fft_padding)
  {
    core::check_plane_fits_int32(dirty_nrow, dirty_ncol, "dirty");
    core::check_plane_fits_int32(raw_psfs.extent(2), raw_psfs.extent(3), "psf");
  }

  context(const context&) = delete;
  context& operator=(const context&) = delete;
  context(context&&) = delete;
  context& operator=(context&&) = delete;

  device_state& state()
  {
    if (state_ == nullptr)
      state_ = std::make_unique<device_state>(exec_device, raw_psfs, xdes, mask, scale_sigmas, dirty_nrow, dirty_ncol,
                                              n_freq, fft_padding);
    return *state_;
  }

 private:
  std::unique_ptr<device_state> state_;
};

struct ddmsc_result {
  std::vector<std::pair<int, int>> peak_coords;
  std::vector<int> scales;
  std::vector<float> gains;
  std::vector<std::vector<float>> coeffs;
  float final_flux = 0.0f;   // peak flux of the mean residual after the last outer iteration
  float stop_flux = 0.0f;    // composed stop-flux threshold used for this call (max of the four limits)
  int total_iterations = 0;  // total minor iterations consumed across all outer cycles
  common::convergence_status status = common::convergence_status::running;  // why the outer loop ended

  ddmsc_result(int max_iter, int coeff_order)
  {
    peak_coords.reserve(max_iter);
    scales.reserve(max_iter);
    gains.reserve(max_iter);
    coeffs.reserve(max_iter);
  };

  void add_component(std::pair<int, int> coords, int scale, float gain)
  {
    peak_coords.push_back(coords);
    scales.push_back(scale);
    gains.push_back(gain);
  };

  void add_coeffs_from_device(core::device_span2d<float> d_coeffs)
  {
    const std::size_t n_components = d_coeffs.extent(0);
    const std::size_t n_order = d_coeffs.extent(1);
    const std::size_t n_total = n_components * n_order;

    std::vector<float> h_buffer(n_total);
    cudaMemcpy(h_buffer.data(), d_coeffs.data_handle(), n_total * sizeof(float), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < n_components; ++i) {
      coeffs.emplace_back(h_buffer.begin() + i * n_order, h_buffer.begin() + (i + 1) * n_order);
    }
  };
};

}  // namespace fast_deconv::algorithm::ddmsc
