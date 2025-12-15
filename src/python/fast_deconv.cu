#include <cuda_runtime.h>

#include <emu/cuda/device/mdspan.hpp>
#include <emu/mdspan.hpp>
#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/matrix/subtract.hpp>
#include <fmt/base.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using stream_resources = fast_deconv::core::stream_resources;

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

template <class MDS>
__global__ void print_mdspan_kernel(MDS m)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
    constexpr int R = (int)MDS::rank();
    printf("mdspan: rank=%d\n", R);

    for (int d = 0; d < R; ++d) {
      printf("extent(%d)=%ld  stride(%d)=%ld\n", d, m.extent(d), d, m.stride(d));
    }
  }

  int rows = (int)m.extent(0);
  int cols = (int)m.extent(1);

  int total = rows * cols;
  if (tid >= total) return;

  int row = tid / cols;
  int col = tid % cols;

  auto val = m(row, col);

  printf("tid=%d row=%d col=%d data=%f\n", tid, row, col, (double)val);
}

template <typename Mdspan>
void print_mdspan(const Mdspan& ms)
{
  fmt::println("{} {} {} {}", ms.extent(0), ms.stride(0), ms.extent(1), ms.stride(1));
  int total   = ms.size();
  int threads = 128;
  int blocks  = (total + threads - 1) / threads;
  print_mdspan_kernel<<<blocks, threads>>>(ms);
  cudaDeviceSynchronize();
}

std::pair<int, float> argmax(const fast_deconv::core::device_span2d_f& data,
                             const fast_deconv::core::device_span2d_b& mask,
                             bool use_abs,
                             stream_resources& resources)
{
  return fast_deconv::matrix::argmax(
    data.data_handle(), mask.data_handle(), data.size(), use_abs, resources);
}

std::pair<int, float> argmax_mdspan(const fast_deconv::core::device_span2d_f& data,
                                    const fast_deconv::core::device_span2d_b& mask,
                                    bool use_abs,
                                    stream_resources& resources)
{
  return fast_deconv::matrix::argmax(data, mask, use_abs, resources);
}

template <typename Mdspan>
void subtract(const Mdspan& A, const Mdspan& B, Mdspan& C, stream_resources& resources)
{
  return fast_deconv::matrix::subtract(
    A.data_handle(), B.data_handle(), C.data_handle(), A.size(), resources);
}

PYBIND11_MODULE(_fast_deconv, m)
{
  m.doc()            = "fast_deconv hello module";
  auto matrix_module = m.def_submodule("matrix", "Matrix module");

  matrix_module.def(
    "argmax", &argmax, R"pbdoc( A function that find the index of the maximum value.)pbdoc");
  matrix_module.def("argmax_mdspan",
                    &argmax_mdspan,
                    R"pbdoc( A function that find the index of the maximum value.)pbdoc");

  matrix_module.def(
    "subtract", &subtract<fast_deconv::core::device_vect_f>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def(
    "subtract", &subtract<fast_deconv::core::device_span2d_f>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def(
    "subtract", &subtract<fast_deconv::core::device_span3d_f>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def(
    "subtract", &subtract<fast_deconv::core::device_span4d_f>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def(
    "subtract", &subtract<fast_deconv::core::device_span5d_f>, R"pbdoc( C = A - B)pbdoc");
  matrix_module.def(
    "subtract", &subtract<fast_deconv::core::device_span6d_f>, R"pbdoc( C = A - B)pbdoc");

  py::class_<fast_deconv::core::stream_resources>(m, "stream_resources").def(py::init<>());

  m.def("inspect", &inspect<fast_deconv::core::device_vect_f>, R"pbdoc(Inspect)pbdoc");
  m.def("inspect", &inspect<fast_deconv::core::device_span2d_f>, R"pbdoc(Inspect)pbdoc");
  // m.def("print", &print_mdspan<fast_deconv::core::device_span2d_f>, R"pbdoc(Print)pbdoc");
  m.def("print", &print_mdspan<fast_deconv::core::device_span2d_fs>, R"pbdoc(Print)pbdoc");
  m.def("print", &print_mdspan<fast_deconv::core::device_span2d_f>, R"pbdoc(Print)pbdoc");
}
