#pragma once

#include <pybind11/pybind11.h>

namespace fast_deconv::python {
// void bind_matrix(pybind11::module_& m);
void bind_wscms(pybind11::module_& m);
void bind_core(pybind11::module_& m);
// void bind_debug(pybind11::module_& m);
// void bind_linalg(pybind11::module_& m);
}  // namespace fast_deconv::python
