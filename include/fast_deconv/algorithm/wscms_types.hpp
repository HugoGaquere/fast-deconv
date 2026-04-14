#pragma once

#include <cufft.h>

#include <fast_deconv/core/span_types.hpp>
#include <vector>

#include "fast_deconv/linalg/detail/fft.cuh"

namespace fast_deconv::algorithm::wscms {

static constexpr int MAX_SPECTRAL_ORDER = 4;

struct WSCMS_ctx {
  core::device_span5d<float> raw_psfs;
  core::device_span2d<float> xdes;
  core::device_span2d<bool> scale_masks;
  core::device_vect<float> scale_sigmas;
  core::host_vect<float> scale_bias;
  core::host_span2d<int> map_pixel_facet;
};

struct WSCMS_params {
  bool clean_negative;
  float peak_factor;
  float gamma;            // CLEAN loop gain
  int max_sub_iteration;  // sub-minor loop iterations per scale selection
  int n_scales;

  // Outer loop parameters
  float stop_flux;                    // stop when peak flux drops below this
  int max_iteration;                  // total minor iterations across all scale selections
  float divergence_factor;            // flux growth ratio that counts as divergence
  float stall_threshold;              // RMS change below this counts as a stall
  std::vector<int> forbidden_scales;  // scales excluded from selection
};

struct scale_convole_ctx {
  int img_nrow, img_ncol;                // image domain size
  int padding_nrow, padding_ncol;        // image domain padding
  int img_padded_nrow, img_padded_ncol;  // image domain padded size
  int freq_nrow, freq_ncol;              // frequency domain size
  int n_batches;
  cufftHandle plan_forward, plan_backward;  // FFT plans

  ~scale_convole_ctx()
  {
    CUFFT_CALL(cufftDestroy(plan_forward));
    CUFFT_CALL(cufftDestroy(plan_backward));
  }
};

struct psf_convolve_ctx {
  int psf_nrow, psf_ncol;                   // PSF spatial size
  int padding_nrow, padding_ncol;           // padding amounts
  int psf_padded_nrow, psf_padded_ncol;     // padded spatial size
  int freq_nrow, freq_ncol;                 // frequency domain size (half-complex)
  int n_batch;                              // batch size (nch per facet)
  cufftHandle plan_forward, plan_backward;  // batched R2C / C2R plans
  cufftHandle plan_backward_2;              // separate C2R plan for conv2

  ~psf_convolve_ctx()
  {
    CUFFT_CALL(cufftDestroy(plan_forward));
    CUFFT_CALL(cufftDestroy(plan_backward));
    CUFFT_CALL(cufftDestroy(plan_backward_2));
  }
};

struct sky_component {
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
};

enum class wscms_exit_reason {
  flux_threshold,  // peak flux dropped below stop_flux
  diverged,        // flux growth exceeded divergence_factor
  stalled,         // all scales stalled (RMS change below threshold)
  max_iterations,  // reached max_iteration count
};

struct wscms_result {
  std::vector<sky_component> components;
  float final_flux;
  int total_iterations = 0;
  wscms_exit_reason exit_reason = wscms_exit_reason::max_iterations;
};

struct scale_selection_result {
  int best_scale;
  int best_row;
  int best_col;
  float best_peak;
};

}  // namespace fast_deconv::algorithm::wscms
