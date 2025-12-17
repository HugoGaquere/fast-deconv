#include <cuda_runtime.h>

#include <emu/cuda/device/mdspan.hpp>
#include <emu/mdspan.hpp>
#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/matrix/subtract.hpp>
#include <fmt/base.h>
#include <fmt/ranges.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace fd_core = fast_deconv::core;
namespace fd_matrix = fast_deconv::matrix;


template <typename Mdspan>
void inspect(const Mdspan& a)
{
  printf("Array data pointer : %p\n", a.data_handle());
  printf("Array dimension : %zu\n", a.rank());
  for (size_t i = 0; i < a.rank(); ++i) {
    printf("Array dimension [%zu] : %zu\n", i, a.extent(i));
    printf("Array stride    [%zu] : %zd\n", i, a.stride(i));
  }
}

template <class Mdspan>
__global__ void print_mdspan_kernel(const Mdspan ms)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  constexpr int R = (int)Mdspan::rank();
  printf("mdspan: rank=%d\n", R);

  auto strides = ms.mapping().strides();
  for (int d = 0; d < R; ++d) {
    printf("extent(%d)=%ld  stride(%d)=%ld\n", d, ms.extent(d), d, ms.stride(d));
    printf("stride(%d)=%ld\n", d, strides[d]);
  }

  for (int i = 0; i < ms.size(); i++) {
    int rows  = ms.extent(0);
    int cols  = ms.extent(1);
    int row   = i / cols;
    int col   = i % cols;
    float val = ms(row, col);
    printf("tid=%d row=%d col=%d data=%f\n", tid, row, col, val);
  }
}

template <typename Mdspan>
void print_mdspan(const Mdspan& ms)
{
  cudaDeviceSynchronize();
  auto ms2 = ms;
  fmt::println("{} {} {} {}", ms2.extent(0), ms2.stride(0), ms2.extent(1), ms2.stride(1));
  fmt::println("{}", ms2.mapping().strides());
  int total   = ms.size();
  int threads = 128;
  int blocks  = (total + threads - 1) / threads;
  print_mdspan_kernel<<<1, 1>>>(ms);
  cudaDeviceSynchronize();
}

PYBIND11_MODULE(_fast_deconv, m)
{
  m.doc()            = "fast_deconv hello module";
  auto matrix_module = m.def_submodule("matrix", "Matrix module");

  matrix_module.def("argmax", &fd_matrix::argmax<fd_core::device_span2d_f, fd_core::device_span2d_b>, R"pbdoc( Argmax.)pbdoc");
  matrix_module.def("argmax", &fd_matrix::argmax<fd_core::device_span2d_fs, fd_core::device_span2d_bs>, R"pbdoc( Argmax.)pbdoc");

  matrix_module.def("subtract", &fd_matrix::subtract<fd_core::device_vect_f>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def("subtract", &fd_matrix::subtract<fd_core::device_span2d_f>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def("subtract", &fd_matrix::subtract<fd_core::device_span3d_f>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def("subtract", &fd_matrix::subtract<fd_core::device_span4d_f>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def("subtract", &fd_matrix::subtract<fd_core::device_span5d_f>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def("subtract", &fd_matrix::subtract<fd_core::device_span6d_f>, R"pbdoc( C = A - B)pbdoc");

  matrix_module.def("subtract", &fd_matrix::subtract<fd_core::device_vect_fs>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def("subtract", &fd_matrix::subtract<fd_core::device_span2d_fs>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def("subtract", &fd_matrix::subtract<fd_core::device_span3d_fs>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def("subtract", &fd_matrix::subtract<fd_core::device_span4d_fs>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def("subtract", &fd_matrix::subtract<fd_core::device_span5d_fs>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def("subtract", &fd_matrix::subtract<fd_core::device_span6d_fs>, R"pbdoc( C = A - B)pbdoc");

  py::class_<fd_core::stream_resources>(m, "stream_resources").def(py::init<>());

  m.def("inspect", &inspect<fd_core::device_vect_f>, R"pbdoc(Inspect)pbdoc");
  m.def("inspect", &inspect<fd_core::device_span2d_f>, R"pbdoc(Inspect)pbdoc");
  // m.def("print", &print_mdspan<fast_deconv::core::device_span2d_f>, R"pbdoc(Print)pbdoc");
  // m.def("print", &print_mdspan<fd_core::device_span2d_fs>, R"pbdoc(Print)pbdoc");
  // m.def("print", &print_mdspan<fast_deconv::core::device_span2d_f>, R"pbdoc(Print)pbdoc");
}
