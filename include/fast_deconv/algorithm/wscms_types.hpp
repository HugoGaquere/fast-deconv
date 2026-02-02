#pragma once

#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::algorithm::wscms {

static constexpr int MAX_SPECTRAL_ORDER = 8;

struct MinorCycleContext {
    // PSFs
    core::span_6d<float> psfs;              // conv_psfs (nfacets, nscales, nch, npol, h, w)
    core::span_6d<float> psfs_2;            // conv2_psfs (nfacets, nscales, nch, npol, h, w)

    // Image metadata
    core::span_4d<float> jones_norm;        // (nch, npol, h, w)
    core::span_2d<float> gains;             // (nfacets, nscales)
    core::span_2d<bool> mask;               // (h, w)
    core::mdspan<int, 1> map_pixels_facets; // pixel -> facet_id, length = h*w

    // Spectral fitting data
    core::span_2d<float> Xdes;              // design matrix (nch, order)
    core::span_1d<float> sqrt_weights;      // sqrt(weights_chan_images) (nch,)
    bool beam_enable;

    // Algorithm parameters
    float peak_factor;
    int n_subminor_iter;
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
