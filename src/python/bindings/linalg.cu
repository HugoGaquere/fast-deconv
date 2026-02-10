#include "fast_deconv_bindings.hpp"

#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/linalg/pinv.cuh>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace fd_core   = fast_deconv::core;
namespace fd_linalg = fast_deconv::linalg;

namespace fast_deconv::python {

void pinv_dispatch(fd_core::device_span2d<float>& A,
                   fd_core::device_span2d<float>& A_pinv,
                   fd_core::stream_resources& resources)
{
  const int rows = static_cast<int>(A.extent(0));
  const int cols = static_cast<int>(A.extent(1));
  fd_linalg::pinv(A, A_pinv, rows, cols, resources);
}

void bind_linalg(py::module_& m)
{
  m.def("pinv", &pinv_dispatch, R"pbdoc(Compute the Moore-Penrose pseudo-inverse of a 2D matrix.

  A is (M x N), A_pinv is (N x M). Uses Cholesky-based computation on GPU.
  Requires M >= N and N <= 16.)pbdoc");
}

}  // namespace fast_deconv::python
