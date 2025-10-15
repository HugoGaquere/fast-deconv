#pragma once
#include <iostream>

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
inline void checkLast(const char *const file, const int line) {
  cudaError_t const err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
