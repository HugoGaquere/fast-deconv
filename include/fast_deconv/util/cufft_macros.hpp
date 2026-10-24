#pragma once
#include <cufft.h>

#include <iostream>
#include <stdexcept>
#include <string>

/// Throws std::runtime_error naming the call and its site when @p val is not CUFFT_SUCCESS.
#define CUFFT_CALL(val) check_cufft((val), #val, __FILE__, __LINE__)
inline void check_cufft(cufftResult status, const char* const func, const char* const file, const int line)
{
  if (status != CUFFT_SUCCESS)
    throw std::runtime_error(std::string("cuFFT error at ") + file + ":" + std::to_string(line) + ": code " +
                             std::to_string(static_cast<int>(status)) + " (" + func + ")");
}

/// CUFFT_CALL for destructors: a throw there would terminate, so the error is only reported.
#define REPORT_CUFFT(val) report_cufft((val), #val, __FILE__, __LINE__)
inline void report_cufft(cufftResult status, const char* const func, const char* const file, const int line) noexcept
{
  if (status != CUFFT_SUCCESS)
    std::cerr << "cuFFT error at " << file << ":" << line << ": code " << static_cast<int>(status) << " (" << func
              << ")" << std::endl;
}
