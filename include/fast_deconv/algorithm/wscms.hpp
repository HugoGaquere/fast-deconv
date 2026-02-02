#pragma once

#include <cuda/std/mdspan>

#include "fast_deconv/algorithm/detail/wscms.hpp"
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>

namespace fast_deconv::algorithm::wscms {

void wscms_minor_cycle(core::span_4d<float> dirty,
                       core::span_4d<float> scaled_dirty,
                       std::uint32_t scale_idx,
                       const MinorCycleContext& ctx,
                       ComponentBuffer& components,
                       core::stream_resources& resources)
{
  detail::wscms_minor_cycle(dirty, scaled_dirty, scale_idx, ctx, components, resources);
}

}  // namespace fast_deconv::algorithm::wscms
