#include "fast_deconv_bindings.hpp"

#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/matrix/subtract.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace fd_core   = fast_deconv::core;
namespace fd_matrix = fast_deconv::matrix;

namespace fast_deconv::python {
void bind_matrix(py::module_& m)
{
  m.def("argmax",
        &fd_matrix::argmax<fd_core::device_vect<float>, fd_core::device_vect<bool>>,
        R"pbdoc( Argmax.)pbdoc");
  m.def("argmax",
        &fd_matrix::argmax<fd_core::device_span2d<float>, fd_core::device_span2d<bool>>,
        R"pbdoc( Argmax.)pbdoc");
  m.def("argmax",
        &fd_matrix::argmax<fd_core::device_span3d<float>, fd_core::device_span3d<bool>>,
        R"pbdoc( Argmax.)pbdoc");
  m.def("argmax",
        &fd_matrix::argmax<fd_core::device_span4d<float>, fd_core::device_span4d<bool>>,
        R"pbdoc( Argmax.)pbdoc");
  m.def("argmax",
        &fd_matrix::argmax<fd_core::device_span5d<float>, fd_core::device_span5d<bool>>,
        R"pbdoc( Argmax.)pbdoc");
  m.def("argmax",
        &fd_matrix::argmax<fd_core::device_span6d<float>, fd_core::device_span6d<bool>>,
        R"pbdoc( Argmax.)pbdoc");

  // Broadcast mask overloads: 2D mask broadcast across leading dimensions
  m.def("argmax",
        &fd_matrix::argmax<fd_core::device_span3d<float>, fd_core::device_span2d<bool>>,
        R"pbdoc( Argmax with 2D broadcast mask.)pbdoc");
  m.def("argmax",
        &fd_matrix::argmax<fd_core::device_span4d<float>, fd_core::device_span2d<bool>>,
        R"pbdoc( Argmax with 2D broadcast mask.)pbdoc");
  m.def("argmax",
        &fd_matrix::argmax<fd_core::device_span5d<float>, fd_core::device_span2d<bool>>,
        R"pbdoc( Argmax with 2D broadcast mask.)pbdoc");
  m.def("argmax",
        &fd_matrix::argmax<fd_core::device_span6d<float>, fd_core::device_span2d<bool>>,
        R"pbdoc( Argmax with 2D broadcast mask.)pbdoc");

  m.def("subtract", &fd_matrix::subtract<fd_core::device_vect<float>>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span2d<float>>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span3d<float>>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span4d<float>>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span5d<float>>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span6d<float>>, R"pbdoc( C = A - B)pbdoc");

  m.def("subtract", &fd_matrix::subtract<fd_core::device_vect_S<float>>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span2d_S<float>>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span3d_S<float>>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span4d_S<float>>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span5d_S<float>>, R"pbdoc( C = A - B)pbdoc");
  m.def("subtract", &fd_matrix::subtract<fd_core::device_span6d_S<float>>, R"pbdoc( C = A - B)pbdoc");
}
}  // namespace fast_deconv::python
