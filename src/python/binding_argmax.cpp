#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include "fast_deconv/argmax.cuh"

namespace nb = nanobind;
using namespace nb::literals;

//std::pair<int, float> _argmax(nb::ndarray<float>& data, nb::ndarray<bool>& mask, bool use_abs) {
//    return fast_deconv::kernels::argmax(data.data(), mask.data(), data.size(), use_abs);
//}

void init_argmax_bindings(nb::module_& m) {
//    m.def("argmax", &_argmax);
}
