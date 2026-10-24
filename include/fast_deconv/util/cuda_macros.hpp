#pragma once
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

/// Throws std::runtime_error naming the call and its site when @p val is not cudaSuccess.
#define CHECK_CUDA(val) check_cuda((val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("CUDA error at ") + file + ":" + std::to_string(line) + ": " +
                             cudaGetErrorString(err) + " (" + func + ")");
}

/// CHECK_CUDA for destructors and noexcept paths: a throw there would terminate, so the error is only reported.
#define REPORT_CUDA(val) report_cuda((val), #val, __FILE__, __LINE__)
inline void report_cuda(cudaError_t err, const char* const func, const char* const file, const int line) noexcept
{
  if (err != cudaSuccess)
    std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(err) << " (" << func << ")"
              << std::endl;
}
