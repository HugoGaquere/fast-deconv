#pragma once

#include "common.hpp"

#include <emu/cuda/device.hpp>
#include <emu/cuda/device/container.hpp>
#include <emu/cuda/memory.hpp>
#include <fast_deconv/algorithm/wscms_op.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>

namespace fast_deconv::example {

void run_subtract_psf_from_dirty()
{
  const auto device_id = emu::cuda::device::current();
  fast_deconv::core::stream_resources resources;

  // 4D array dimensions: (nch, npol, height, width)
  constexpr size_t nch    = 1;
  constexpr size_t npol   = 1;
  constexpr size_t height = 10000;
  constexpr size_t width  = 10000;
  constexpr size_t total  = nch * npol * height * width;

  // Allocate device memory
  auto psf_data    = emu::cuda::device::make_container<float>(device_id, total);
  auto dirty_data  = emu::cuda::device::make_container<float>(device_id, total);
  auto out_data    = emu::cuda::device::make_container<float>(device_id, total);
  auto coeffs_data = emu::cuda::device::make_container<float>(device_id, nch);

  // Fill with random data
  fill_device_random(psf_data.data(), total);
  fill_device_random(dirty_data.data(), total);
  fill_device_random(coeffs_data.data(), nch);

  // Create strided mdspan views (row-major strides)
  using extents_t = emu::dextents<std::size_t, 4>;
  using mapping_t = emu::layout_stride::mapping<extents_t>;

  extents_t exts(nch, npol, height, width);
  std::array<std::size_t, 4> strides = {
    npol * height * width,  // stride for dim 0
    height * width,         // stride for dim 1
    width,                  // stride for dim 2
    1                       // stride for dim 3
  };
  mapping_t mapping(exts, strides);

  core::device_span4d_S<float> psf(psf_data.data(), mapping);
  core::device_span4d_S<float> dirty(dirty_data.data(), mapping);
  core::device_span4d_S<float> out(out_data.data(), mapping);

  // 1D span for coefficients (layout_right)
  core::device_vect<float> coeffs(coeffs_data.data(), nch);

  float gain = 0.1f;

  // // Warmup
  // algo::wscms::subtract_psf_from_dirty_async(psf, dirty, coeffs, out, gain, resources);
  // resources.sync();

  // Benchmark run (for nsight-compute profiling)
  algo::wscms::subtract_psf_from_dirty_async(psf, dirty, coeffs, out, gain, resources);
  resources.sync();
}

}  // namespace fast_deconv::example
