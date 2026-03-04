#pragma once

#include <cufft.h>

#include <fast_deconv/core/span_types.hpp>

#include "fast_deconv/linalg/detail/fft.cuh"

namespace fast_deconv::algorithm::wscms {

static constexpr int MAX_SPECTRAL_ORDER = 8;

struct MinorCycleContext {
  // Image metadata
  core::device_span4d<float> jones_norm;     // (nch, npol, h, w)
  core::host_span2d<int> map_pixels_facets;  // pixel -> facet_id

  // Spectral fitting data
  core::device_span2d<float> Xdes;        // design matrix (nch, order)
  core::device_vect<float> sqrt_weights;  // sqrt(weights_chan_images) (nch,)
  bool beam_enable;

  // Algorithm parameters
  float peak_factor;
  uint n_subminor_iter;
  bool do_abs;
};

struct ComponentEntry {
  int x;
  int y;
  int scale_idx;
  float gain;
  float coeffs[MAX_SPECTRAL_ORDER];
  int n_coeffs;
};

struct ComponentBuffer {
  ComponentEntry* entries;  // host-side pre-allocated array
  int count;                // filled by minor_cycle
  int capacity;             // = n_subminor_iter
};

struct scale_selection_result {
  int best_scale;
  int best_x;
  int best_y;
  float best_peak;
};

struct scale_convole_ctx {
  int img_x, img_y;                // image domain size
  int padding_x, padding_y;        // image_domain padding
  int img_padded_x, img_padded_y;  // image domain padded size
  int freq_x, freq_y;              // frequency domain size
  int n_batches;
  cufftHandle plan_forward, plan_backward;  // FFT plans

  ~scale_convole_ctx()
  {
    CUFFT_CALL(cufftDestroy(plan_forward));
    CUFFT_CALL(cufftDestroy(plan_backward));
  }
};

}  // namespace fast_deconv::algorithm::wscms
