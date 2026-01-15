#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/matrix/subtract.hpp>
#include <pybind11/pybind11.h>

#include "fast_deconv_bindings.hpp"

namespace py = pybind11;

namespace fd_core = fast_deconv::core;
namespace fd_matrix = fast_deconv::matrix;

namespace fast_deconv::python
{
void bind_matrix(py::module_& m)
{
  m.def("argmax", &fd_matrix::argmax<fd_core::device_vect_f, fd_core::device_vect_b>, R"pbdoc( Argmax.)pbdoc");
  m.def("argmax", &fd_matrix::argmax<fd_core::device_span2d_f, fd_core::device_span2d_b>, R"pbdoc( Argmax.)pbdoc");
  m.def("argmax", &fd_matrix::argmax<fd_core::device_span3d_f, fd_core::device_span3d_b>, R"pbdoc( Argmax.)pbdoc");
  m.def("argmax", &fd_matrix::argmax<fd_core::device_span4d_f, fd_core::device_span4d_b>, R"pbdoc( Argmax.)pbdoc");
  m.def("argmax", &fd_matrix::argmax<fd_core::device_span5d_f, fd_core::device_span5d_b>, R"pbdoc( Argmax.)pbdoc");
  m.def("argmax", &fd_matrix::argmax<fd_core::device_span6d_f, fd_core::device_span6d_b>, R"pbdoc( Argmax.)pbdoc");

  m.def("subtract", &fd_matrix::subtract<fd_core::device_vect_f>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span2d_f>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span3d_f>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span4d_f>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span5d_f>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span6d_f>, R"pbdoc( C = A - B)pbdoc");

  m.def("subtract", &fd_matrix::subtract<fd_core::device_vect_fs>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span2d_fs>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span3d_fs>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span4d_fs>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span5d_fs>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span6d_fs>, R"pbdoc( C = A - B)pbdoc");
}
}  // namespace fast_deconv::python
