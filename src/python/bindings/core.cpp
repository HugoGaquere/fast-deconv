#include <pybind11/pybind11.h>

#include <fast_deconv/core/resources.hpp>

#include "fast_deconv_bindings.hpp"

namespace py = pybind11;


namespace fast_deconv::python {

void bind_core(py::module_& m)
{
  py::class_<fast_deconv::core::resources>(m, "Resources");
}

}  // namespace fast_deconv::python
