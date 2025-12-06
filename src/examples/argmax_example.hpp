#pragma once

#include "common.hpp"

#include <emu/cuda/device.hpp>
#include <emu/cuda/device/container.hpp>
#include <emu/cuda/memory.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/argmax.hpp>

namespace fast_deconv::example {
void run_argmax()
{
  const auto device_id = emu::cuda::device::current();
  fast_deconv::core::stream_resources resources;

  const size_t N = 10000;

  auto data = emu::cuda::device::make_container<float>(device_id, N);
  auto mask = emu::cuda::device::make_container<bool>(device_id, N);

  fill_device_random(data.data(), N);
  fill_device_bool(mask.data(), N);

  std::pair<int, float> res =
    fast_deconv::matrix::argmax(data.data(), mask.data(), N, false, resources);

  fmt::println("Argmax {}", res);
}

}  // namespace fast_deconv::example
