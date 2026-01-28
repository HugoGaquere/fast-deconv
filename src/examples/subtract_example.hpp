#pragma once

#include "common.hpp"

#include <emu/cuda/device.hpp>
#include <emu/cuda/device/mdcontainer.hpp>
#include <emu/cuda/memory.hpp>
#include <emu/cuda/device/mdspan.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/subtract.hpp>

namespace fast_deconv::example {
void run_subtract()
{
  // Allocate device memory
  size_t rows = 4, cols = 6;
  size_t total = rows * cols;

  // Creates and owns device memory
  std::vector<float> data_a(total, 1.0);
  std::vector<float> data_b(total, 1.0);
  std::vector<float> data_c(total, 1.0);

  emu::cuda::device::mdspan_2d<float> A(data_a.data(), rows, cols);
  emu::cuda::device::mdspan_2d<float> B(data_b.data(), rows, cols);
  emu::cuda::device::mdspan_2d<float> C(data_c.data(), rows, cols);


  // fill_device_random(A.data(), N);
  // fill_device_random(B.data(), N);

  fast_deconv::core::stream_resources resources;
  fast_deconv::matrix::subtract_async(A, B, C, resources);
  resources.sync();
}

}  // namespace fast_deconv::example
