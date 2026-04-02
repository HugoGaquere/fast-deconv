#pragma once
#include "cublas_v2.h"

#include <iostream>

#define CHECK_CUBLAS(val) check_cublas((val), #val, __FILE__, __LINE__)
inline void check_cublas(cublasStatus_t status,
                         const char* const func,
                         const char* const file,
                         const int line)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS Runtime Error at: " << file << ":" << line << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
