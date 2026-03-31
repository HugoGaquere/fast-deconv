#pragma once

#include <cufft.h>

#include <fast_deconv/core/span_types.hpp>
#include <vector>

#include "fast_deconv/linalg/detail/fft.cuh"

namespace fast_deconv::algorithm::wscms {

static constexpr int MAX_SPECTRAL_ORDER = 4;

struct WSCMS_ctx {
  core::device_span4d<float> jones_norm;
  core::device_span2d<float> xdes;
  core::device_vect<float> weights_freq;
  core::device_span2d<bool> scale_masks;
  core::device_vect<float> scale_sigmas;
  core::host_vect<float> scale_bias;
  core::host_span2d<int> map_pixel_facet;
  core::host_span2d<float> gains;
};

struct WSCMS_params {
  bool beam_enable;
  bool do_abs;
  bool per_scale_mask;
  float peak_factor;
  int max_subminor_iter;
  int n_scales;
  float padding;
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

struct sky_component {
  int row;
  int col;
  int scale_idx;
  float gain;
  std::vector<float> coeffs;
};

// struct ComponentBuffer {
//   ComponentEntry* entries;  // host-side pre-allocated array
//   int count;                // filled by minor_cycle
//   int capacity;             // = n_subminor_iter
// };

struct scale_selection_result {
  int best_scale;
  int best_row;
  int best_col;
  float best_peak;
};

}  // namespace fast_deconv::algorithm::wscms
