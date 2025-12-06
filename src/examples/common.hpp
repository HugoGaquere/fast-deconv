#pragma once

#include <cuda_runtime.h>

#include <fast_deconv/util/cuda_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

std::vector<float> random_vector(std::size_t n)
{
  std::mt19937 rng{std::random_device{}()};
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> v(n);
  std::generate(v.begin(), v.end(), [&] { return dist(rng); });
  return v;
}

void fill_device_random(float* ptr, std::size_t n)
{
  auto h_vect = random_vector(n);
  CHECK_CUDA(cudaMemcpy(ptr, h_vect.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

void fill_device_bool(bool* ptr, std::size_t n, bool value=true) {
  cudaMemset(ptr, static_cast<int>(value), n * sizeof(bool));
}
