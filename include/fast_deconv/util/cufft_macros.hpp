#pragma once
#include <cufft.h>

#include <cstdlib>
#include <iostream>

#define CUFFT_CALL(val) check_cufft((val), #val, __FILE__, __LINE__)
inline void check_cufft(cufftResult status, const char* const func, const char* const file, const int line)
{
  if (status != CUFFT_SUCCESS) {
    std::cerr << "cuFFT Error at: " << file << ":" << line << std::endl;
    std::cerr << "code (" << static_cast<int>(status) << ") " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
