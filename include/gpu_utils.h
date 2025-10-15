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
