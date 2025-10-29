#pragma once

#include <cuda/std/mdspan>

#include "fast_deconv/algorithm/detail/wscms.hpp"
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>

namespace fast_deconv::algorithm::wscms {

void wscms_minor_cycle(core::span_4d<float> dirty,
                       core::span_4d<float> scaled_dirty,
                       core::span_2d<bool> mask,
                       core::span_6d<float> psfs,
                       core::span_4d<float> jones_norm,
                       core::span_2d<float> gains,
                       std::uint32_t scale_idx,
                       Facets facets,
                       Params params,
                       core::stream_resources& resources)
{
  // Run differents checks
  detail::wscms_minor_cycle(dirty, scaled_dirty, mask, psfs, jones_norm, gains, scale_idx, facets, params, resources);
}

}  // namespace fast_deconv::algorithm::wscms

// Notes:
// For now, we asssume 1 channel and 1 polarization

// namespace fast_deconv::wscms {
//
// template<typename T>
// struct Facets {
//     std::vector<std::vector<T>> edges;
//     std::vector<float2> centers;
// };
//
// //  components_center: the (l,m) centre of the component in pixels
// //  sols: Nd array of coeffs with length equal to the number of basis functions representing the
// component.
// //  scale_idx: the scale index
// //  gains: clean loops gains
// struct SubminorLoopResults {
//     // std::vector<std::pair<uint, uint>> components_center;
//     std::vector<float> sols;
//     std::vector<int> scales_idx;
//     std::vector<float> gains;
// };
//
// class WSCMS {
// public:
//     WSCMS() {};
//     WSCMS(Facets<float> facets, float *PSFs) : _facets(facets), _PSFs(PSFs) {};
//     SubminorLoopResults  run_subminor_loop(float *dirty, float *scaled_dirty, bool *mask,
//     unsigned int scale_idx);
// private:
//     Facets<float> _facets;
//     float *_PSFs;
//     std::vector<float> _gains;
//     float2 _cell_size_radian;
//     float _peak_factor;
//     float _gamma;
// };
// }
