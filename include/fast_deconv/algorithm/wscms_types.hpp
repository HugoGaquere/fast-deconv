#pragma once

#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::algorithm::wscms {

static constexpr int MAX_SPECTRAL_ORDER = 8;

struct MinorCycleContext {
  // Image metadata
  core::device_span4d<float> jones_norm;         // (nch, npol, h, w)
  core::host_span2d<int> map_pixels_facets;   // pixel -> facet_id

  // Spectral fitting data
  core::device_span2d<float> Xdes;          // design matrix (nch, order)
  core::device_vect<float> sqrt_weights;    // sqrt(weights_chan_images) (nch,)
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

}  // namespace fast_deconv::algorithm::wscms
