#include <fast_deconv/core/stream_resources.hpp>
#include <pybind11/pybind11.h>

#include "fast_deconv_bindings.hpp"

namespace py = pybind11;

namespace fd_core = fast_deconv::core;

namespace fast_deconv::python
{
void bind_core(py::module_& m)
{
  py::class_<fd_core::stream_resources>(m, "stream_resources").def(py::init<>());
}
}  // namespace fast_deconv::python
