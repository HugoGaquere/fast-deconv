#pragma once

#include <cuda_runtime.h>

#include <curand.h>
#include <fast_deconv/util/cuda_macros.hpp>

#include <cstddef>

void fill_device_random(float* ptr, std::size_t n)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  curandGenerateUniform(gen, ptr, n);
  curandDestroyGenerator(gen);
}

void fill_device_bool(bool* ptr, std::size_t n, bool value = true)
{
  cudaMemset(ptr, static_cast<int>(value), n * sizeof(bool));
}
