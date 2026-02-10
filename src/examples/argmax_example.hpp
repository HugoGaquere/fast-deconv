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

  const size_t n = 10000 * 10000;

  auto data = emu::cuda::device::make_container<float>(device_id, n);
  auto mask = emu::cuda::device::make_container<bool>(device_id, n);

  fill_device_random(data.data(), n);
  fill_device_bool(mask.data(), n);

  std::pair<int, float> res =
    fast_deconv::matrix::argmax(data.data(), mask.data(), n, n, false, resources);

  fmt::println("Argmax {}", res);
}

}  // namespace fast_deconv::example
