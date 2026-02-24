#include "bindings/fast_deconv_bindings.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace fd_py = fast_deconv::python;

PYBIND11_MODULE(_fast_deconv, m)
{
  m.doc()            = "fast_deconv hello module";
  auto matrix_module = m.def_submodule("matrix", "Matrix module");
  auto wscms_module  = m.def_submodule("wscms", "WSCMS module");
  auto linalg_module = m.def_submodule("linalg", "Linear algebra module");

  fd_py::bind_core(m);
  fd_py::bind_wscms(wscms_module);
  // fd_py::bind_matrix(matrix_module);
  // fd_py::bind_linalg(linalg_module);
  // fd_py::bind_debug(m);
}
