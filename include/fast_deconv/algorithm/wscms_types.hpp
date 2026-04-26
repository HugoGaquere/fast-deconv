#pragma once

#include <cufft.h>

#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <vector>

#include "fast_deconv/linalg/fft.hpp"

namespace fast_deconv::algorithm::wscms {

static constexpr int MAX_SPECTRAL_ORDER = 4;

/// Image-domain scale convolution context: one R2C + one C2R plan over the padded image.
struct scale_convolve_ctx : public linalg::convolve_ctx {
  scale_convolve_ctx() = default;

  /// Build plans for an unbatched padded image transform and bind a workspace from the pool.
  scale_convolve_ctx(const core::resources& resources, int nrow, int ncol, float padding)
      : linalg::convolve_ctx(nrow, ncol, /*plan_batch=*/1, /*n_backward_plans=*/1, padding)
  {
    const auto& stream_r = resources.get_stream_resources();
    void* work = resources.alloc_async<void>(work_size, stream_r);
    stream_r.sync();
    set_work_area(work);
  }

  cufftHandle& plan_forward() { return plans_forward[0]; }
  cufftHandle plan_forward() const { return plans_forward[0]; }
  cufftHandle& plan_backward() { return plans_backward[0]; }
  cufftHandle plan_backward() const { return plans_backward[0]; }
};

/// PSF-domain convolution context: batched R2C + two C2R plans (one for conv, one for conv^2).
struct psf_convolve_ctx : public linalg::convolve_ctx {
  psf_convolve_ctx() = default;

  /// Build batched plans (over n_freq channels) and bind a workspace from the pool.
  psf_convolve_ctx(const core::resources& resources, int psf_nrow, int psf_ncol, int nch, float padding)
      : linalg::convolve_ctx(psf_nrow, psf_ncol, /*plan_batch=*/nch, /*n_backward_plans=*/2, padding)
  {
    const auto& stream_r = resources.get_stream_resources();
    void* work = resources.alloc_async<void>(work_size, stream_r);
    stream_r.sync();
    set_work_area(work);
  }

  cufftHandle& plan_forward() { return plans_forward[0]; }
  cufftHandle plan_forward() const { return plans_forward[0]; }
  cufftHandle& plan_backward() { return plans_backward[0]; }
  cufftHandle plan_backward() const { return plans_backward[0]; }
  cufftHandle& plan_backward_2() { return plans_backward[1]; }
  cufftHandle plan_backward_2() const { return plans_backward[1]; }
};

struct workspace {
  scale_convolve_ctx scale_convolve;
  psf_convolve_ctx psf_convolve;
  core::device_span4d<float> raw_psfs;
  core::device_span2d<float> xdes;
  core::device_span2d<bool> mask;
  core::device_vect<float> scale_sigmas;
  core::host_vect<float> scale_bias;
  core::host_span2d<int> map_pixel_facet;
};

struct params {
  int max_iteration;            // total minor iterations across all scale selections
  float stop_flux_threshold;    // stop when peak flux drops below this
  float divergence_factor;      // flux growth ratio that counts as divergence
  float scale_stall_threshold;  // RMS change below this counts as a stall

  bool clean_negative;
  float peak_factor;
  float gamma;              // CLEAN loop gain
  int max_clean_iteration;  // sub-minor loop iterations per scale selection
};

struct context {
  core::resources exec_resources;
  wscms::workspace workspace;

  /**
   * @brief Build the GPU resource pool, the cuFFT plans for scale and PSF
   *        convolutions, and bind the user-provided spans into the workspace.
   *
   * Spans are stored as views — the caller must keep their backing memory alive
   * for the lifetime of the context.
   */
  context(int exec_device, const core::device_span4d<float>& raw_psfs, const core::device_span2d<float>& xdes,
          const core::device_span2d<bool>& mask, const core::device_vect<float>& scale_sigmas,
          const core::host_vect<float>& scale_bias, const core::host_span2d<int>& map_pixel_facet, int dirty_nrow,
          int dirty_ncol, int n_freq, float fft_padding)
      : exec_resources(exec_device),
        workspace{
            .scale_convolve = scale_convolve_ctx(exec_resources, dirty_nrow, dirty_ncol, fft_padding),
            .psf_convolve = psf_convolve_ctx(exec_resources, static_cast<int>(raw_psfs.extent(2)),
                                             static_cast<int>(raw_psfs.extent(3)), n_freq, fft_padding),
            .raw_psfs = raw_psfs,
            .xdes = xdes,
            .mask = mask,
            .scale_sigmas = scale_sigmas,
            .scale_bias = scale_bias,
            .map_pixel_facet = map_pixel_facet,
        }
  {
  }
};

struct wscms_result {
  std::vector<std::pair<int, int>> peak_coords;
  std::vector<int> scales;
  std::vector<float> gains;
  std::vector<std::vector<float>> coeffs;
  float final_flux = 0.0f;   // peak flux of the mean residual after the last outer iteration
  int total_iterations = 0;  // total minor iterations consumed across all outer cycles

  wscms_result(int max_iter, int coeff_order)
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

/* struct WSCMS_ctx {
  core::device_span4d<float> raw_psfs;
  core::device_span2d<float> xdes;
  core::device_span2d<bool> scale_mask;
  core::device_vect<float> scale_sigmas;
  core::host_vect<float> scale_bias;
  core::host_span2d<int> map_pixel_facet;
}; */

// struct WSCMS_params {
//   bool clean_negative;
//   float peak_factor;
//   float gamma;            // CLEAN loop gain
//   int max_sub_iteration;  // sub-minor loop iterations per scale selection
//   int n_scales;

//   // Outer loop parameters
//   float stop_flux;                    // stop when peak flux drops below this
//   int max_iteration;                  // total minor iterations across all scale selections
//   float divergence_factor;            // flux growth ratio that counts as divergence
//   float stall_threshold;              // RMS change below this counts as a stall
//   std::vector<int> forbidden_scales;  // scales excluded from selection
// };

/* struct wscms_resources {
  psf_convolve_ctx psf_convolve_ctx;
  scale_convolve_ctx scale_convolve_ctx;
} */
;

/* struct sky_component {
  int row;
  int col;
  int scale_idx;
  float gain;
  std::vector<float> coeffs;
};

/// @brief Host-side metadata for a single sky component (coefficients stored separately on device).
struct component_meta {
  int row;
  int col;
  int scale_idx;
  float gain;
} */
;

// enum class wscms_exit_reason {
//   flux_threshold,  // peak flux dropped below stop_flux
//   diverged,        // flux growth exceeded divergence_factor
//   stalled,         // all scales stalled (RMS change below threshold)
//   max_iterations,  // reached max_iteration count
// };

// struct wscms_result_old {
//   std::vector<sky_component> components;
//   float final_flux;
//   int total_iterations = 0;
//   wscms_exit_reason exit_reason = wscms_exit_reason::max_iterations;
// };

/* struct scale_selection_result {
  int best_scale;
  int best_row;
  int best_col;
  float best_peak;
};
 */

}  // namespace fast_deconv::algorithm::wscms
