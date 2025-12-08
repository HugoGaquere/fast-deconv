#include <emu/cuda/device/mdspan.hpp>
#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/matrix/subtract.hpp>
#include <fmt/base.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using mdspan_2df_device = emu::cuda::device::mdspan_2d<float>;
using mdspan_2db_device = emu::cuda::device::mdspan_2d<bool>;
using stream_resources  = fast_deconv::core::stream_resources;

std::pair<int, float> argmax(const mdspan_2df_device& data,
                             const mdspan_2db_device& mask,
                             bool use_abs,
                             stream_resources& resources)
{
  return fast_deconv::matrix::argmax(
    data.data_handle(), mask.data_handle(), data.size(), use_abs, resources);
}

void subtract(const mdspan_2df_device& A,
              const mdspan_2df_device& B,
              mdspan_2df_device& C,
              stream_resources& resources)
{
  return fast_deconv::matrix::subtract(
    A.data_handle(), B.data_handle(), C.data_handle(), A.size(), resources);
}

PYBIND11_MODULE(_fast_deconv, m)
{
  m.doc() = "fast_deconv hello module";
  auto matrix_module = m.def_submodule("matrix", "Matrix module");

  matrix_module.def(
    "argmax", &argmax, R"pbdoc( A function that find the index of the maximum value.)pbdoc");

  matrix_module.def("subtract", &subtract, R"pbdoc( C = A - B)pbdoc");

  py::class_<fast_deconv::core::stream_resources>(m, "stream_resources").def(py::init<>());
}
