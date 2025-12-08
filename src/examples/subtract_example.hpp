#pragma once

#include "common.hpp"

#include <emu/cuda/device.hpp>
#include <emu/cuda/device/container.hpp>
#include <emu/cuda/memory.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/subtract.hpp>

namespace fast_deconv::example {
void run_subtract()
{
  const auto device_id = emu::cuda::device::current();
  fast_deconv::core::stream_resources resources;

  const size_t N = 10000*10000;

  auto A = emu::cuda::device::make_container<float>(device_id, N);
  auto B = emu::cuda::device::make_container<float>(device_id, N);
  auto C = emu::cuda::device::make_container<float>(device_id, N);

  fill_device_random(A.data(), N);
  fill_device_random(B.data(), N);

  fast_deconv::matrix::subtract_async(A.data(), B.data(), C.data(), N, resources);

  resources.sync();
}

}  // namespace fast_deconv::example
