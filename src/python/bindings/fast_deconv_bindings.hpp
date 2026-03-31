#pragma once

#include <pybind11/pybind11.h>

namespace fast_deconv::python {
void bind_wscms(pybind11::module_& m);
}  // namespace fast_deconv::python
