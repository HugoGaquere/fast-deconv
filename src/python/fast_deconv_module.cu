#include <pybind11/pybind11.h>

#include "bindings/fast_deconv_bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_fast_deconv, m)
{
  m.doc() = "fast_deconv ddmsc module";
  fast_deconv::python::bind_ddmsc(m);
}
