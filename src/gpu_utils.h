#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

template<typename T>
using GpuVec = nb::ndarray<T, nb::ndim<1>, nb::device::cuda, nb::c_contig>;
template<typename T>
using Gpu2D = nb::ndarray<T, nb::ndim<2>, nb::device::cuda, nb::c_contig>;
template<typename T>
using Gpu3D = nb::ndarray<T, nb::ndim<3>, nb::device::cuda, nb::c_contig>;


#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
inline void check_last(const char *const file, const int line) {
  cudaError_t const err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
