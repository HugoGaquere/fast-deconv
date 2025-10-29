#pragma once
#include <cuda/std/mdspan>

#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/matrix/argmax.cuh>

#include <cstdio>
#include <vector>

namespace fast_deconv::algorithm::wscms::detail {

// Typical dirty fron DDFacet (freq, pol, height, width) => float32
// PSFs (facet, scale, freq, pol, heigt, width) => float32

uint find_facet_idx(
  Facets facets, uint dirty_width, uint dirty_height, uint peak_x, uint peak_y, Params params)
{
  uint facet_idx = 0;
  float min_dist = std::numeric_limits<float>::max();
  for (uint i = 0; i < facets.centers.size(); i++) {
    float l = params.cell_size_radian.x * (peak_x - static_cast<float>(dirty_width) / 2);
    float m = params.cell_size_radian.y * (peak_y - static_cast<float>(dirty_height) / 2);
    float2 facet_center = facets.centers[i];
    float distance      = std::sqrt((l - facet_center.x) * (l - facet_center.x) +
                               (m - facet_center.y) * (m - facet_center.y));
    if (distance < min_dist) {
      facet_idx = i;
      min_dist  = distance;
    }
  }
  return facet_idx;
}

inline std::pair<int, float> find_peak(core::Span4d<float> cube,
                                       core::Span2d<bool> mask,
                                       uint dirty_width,
                                       uint dirty_height,
                                       core::stream_resources& resources)
{
  auto data_view = core::make_span(cube.data_handle(), core::Extents2D(dirty_width, dirty_height));
  return fast_deconv::matrix::argmax(
    data_view.data_handle(), mask.data_handle(), data_view.size(), false, resources);
}

void wscms_minor_cycle(core::Span4d<float> dirty,
                       core::Span4d<float> scaled_dirty,
                       core::Span2d<bool> mask,
                       core::Span6d<float> psfs,
                       core::Span4d<float> jones_norm,
                       core::Span2d<float> gains,
                       std::uint32_t scale_idx,
                       Facets facets,
                       Params params,
                       core::stream_resources& resources)
{
  // 1. Find peak x,y
  // 2. Load Facet / PSF at pos x,y
  // 3. Compute sky model component
  // 4. Compute aligned patch edges
  // 5. Clean dirty
  //   a. compute flux_scaled_dirty
  //   b. subtract flux scaled_dirty from dirty
  // 6. Clean Scaled Dirty
  //   a. scaled_dirty - conv2_PSF

  size_t nof_elements = dirty.size();
  uint n_frequencies  = dirty.extent(0);
  uint n_pol          = dirty.extent(1);
  uint dirty_width    = dirty.extent(2);
  uint dirty_height   = dirty.extent(3);
  uint n_facets       = psfs.extent(0);
  uint n_scales       = psfs.extent(1);
  uint psf_width      = psfs.extent(4);
  uint psf_height     = psfs.extent(5);

  auto max_dirty   = find_peak(dirty, mask, dirty_width, dirty_height, resources);
  size_t peak_idx  = max_dirty.first;
  float peak_value = max_dirty.second;

  float threshold = params.peak_factor * peak_value;
  // Update the mask where scaled_dirty > threshold and mask==False
  // We do not follow Cyril's convention (searching for peaks where mask==False)
  // Here we search for peaks where mask==True
  // FIXME: MAYBE, we do not need to update the mask

  uint n_iter = 0;
  while (peak_value > threshold && n_iter < params.max_iter) {
    uint peak_x = peak_idx / dirty_height;
    uint peak_y = peak_idx % dirty_width;

    //  Find facet idx
    uint facet_idx = find_facet_idx(facets, dirty_width, dirty_height, peak_x, peak_y, params);
    // Get the corresponding gain
    float gain = gains(facet_idx, scale_idx);

    // 3. Compute offset until the corresponding psf
    // uint psf_offset = facet_idx * psf_width * psf_height;
    // auto psf_view = core::make_span(psfs.data_handle() + psf_offset, core::Extents2D(psf_width, psf_height));

    if (n_frequencies > 1) {
      // Get peak and jones norm for each frequency
      std::vector<float> peak_per_freq;
      std::vector<float> jn_per_freq;
      for (int i = 0; i < n_frequencies; i++) {
        peak_per_freq.push_back(dirty(i, 0, peak_x, peak_y));
        jn_per_freq.push_back(jones_norm(i, 0, peak_x, peak_y));
      }
    }

    // Subtract psf from dirty
    // Scale each coeff by gain
    // Scale each psf freq by scaled coeff
    

    n_iter++;
  }
}

}  // namespace fast_deconv::algorithm::wscms::detail
// SubminorLoopResults WSCMS::run_subminor_loop(float* dirty,
//                                              float* scaled_dirty,
//                                              bool* mask,
//                                              unsigned int scale_idx)
// {
// ===== SUMARRY =====
// 1. Find peak x,y
// 2. Load Facet / PSF at pos x,y
// 3. Compute sky model component
// 4. Compute aligned patch edges
// 5. Clean dirty
//   a. compute flux_scaled_dirty
//   b. subtract flux scaled_dirty from dirty
// 6. Clean Scaled Dirty
//   a. scaled_dirty - conv2_PSF

// SubminorLoopResults results;
//    uint2 psf_shape{static_cast<uint>(this->_PSFs.shape(0)),
//    static_cast<uint>(this->_PSFs.shape(1))};
//
//    // TODO: Add do_absolute argument
//    size_t nof_elements = dirty.size();
//
//    auto max_dirty = argmax(dirty.data(), mask.data(), nof_elements, false);
//    size_t peak_idx = max_dirty.first;
//    float peak_value = max_dirty.second;
//    printf("Dirty max_dirty = %f at %d/n",
//           max_dirty.second, max_dirty.first);
//
//    float threshold = this->_peak_factor * peak_value;
//    // Update the mask where scaled_dirty > threshold and mask==False
//    // We do not follow Cyril's convention (searching for peaks where mask==False)
//    // Here we search for peaks where mask==True
//    //
//    // MAYBE, we do not need to update the mask
//
//    uint n_iter = 0;
//    uint max_iter = 1000; // TODO: pass it as an arguments
//    while (peak_value > threshold && n_iter < max_iter) {
//
//        uint peak_x = peak_idx / dirty.shape(1);
//        uint peak_y = peak_idx % dirty.shape(0);
//
//        // 2. Find facet idx
//        uint facet_idx = 0;
//        float min_dist = std::numeric_limits<float>::max();
//        for (uint i; i < this->_facets.centers.size(); i++) {
//            float l = this->_cell_size_radian.x * (peak_x - static_cast<float>(dirty.shape(0)) /
//            2); float m = this->_cell_size_radian.y * (peak_y -
//            static_cast<float>(dirty.shape(1)) / 2); float2 facet_center =
//            this->_facets.centers[i]; float distance = std::sqrt((l - facet_center.x) * (l -
//            facet_center.x) + (m - facet_center.y) * (m - facet_center.y)); if (distance <
//            min_dist) {
//                facet_idx = i;
//                min_dist = distance;
//            }
//        }
//
//        float gain = this->_gains[facet_idx];
//        float coeffs = peak_value;
//
//        // 3. Save sky model component
//        results.components_center.push_back({peak_x, peak_y});
//        results.sols.push_back(peak_value);
//        results.scales_idx.push_back(scale_idx);
//        results.gains.push_back(gain);
//
//        auto [dirty_patch, psf_patch] = compute_aligned_patch_edges(
//            peak_x, peak_y, dirty.shape(0), dirty.shape(1), psf_shape.x, psf_shape.y);
//
//        // Clean dirty image
//        uint offset = facet_idx * psf_shape.x * psf_shape.y;
//        float *psf = this->_PSFs.data() + offset;
//
//        // TODO: we need to convolve the psf x kernel
//
//
//        cudaStream_t stream;
//        cudaStreamCreate(&stream);
//        clean_dirty_async(dirty.data(), psf, dirty.shape(1), psf_shape.y, dirty_patch,
//        psf_patch, gain, coeffs, stream);
//
//        // Clean Scaled image
//
//
//    }

//   return results;
// }

// }  // namespace fast_deconv::algorithm::detail
