#pragma once

// #include "fast_deconv/algorithm/detail/wscms_old.hpp"
#include <cuda/std/mdspan>

#include <vector>

#include <fast_deconv/algorithm/detail/wscms_minor_loop.cuh>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>

namespace fast_deconv::algorithm::wscms {

std::vector<ComponentEntry> wscms_minor_cycle(core::device_span4d<float> dirty,
                       core::device_span4d<float> scaled_dirty,
                       core::device_span6d<float> psfs,
                       core::device_span6d<float> psfs_2,
                       core::device_span4d<bool> mask,
                       core::host_span2d<float> gains,
                       std::uint32_t scale_idx,
                       const MinorCycleContext& ctx)
{
  return detail::wscms_minor_cycle(
    dirty, scaled_dirty, psfs, psfs_2, mask, gains, scale_idx, ctx);
}

}  // namespace fast_deconv::algorithm::wscms
