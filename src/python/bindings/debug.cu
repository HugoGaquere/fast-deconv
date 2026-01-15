#include <cuda_runtime.h>

#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fmt/base.h>
#include <fmt/ranges.h>
#include <pybind11/pybind11.h>

#include "fast_deconv_bindings.hpp"

namespace py = pybind11;

namespace fd_core = fast_deconv::core;

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
  print_mdspan_kernel<<<1, 1>>>(ms);
  cudaDeviceSynchronize();
}

namespace fast_deconv::python
{
void bind_debug(py::module_& m)
{
  m.def("inspect", &inspect<fd_core::device_vect_f>, R"pbdoc(Inspect)pbdoc");
  m.def("inspect", &inspect<fd_core::device_span2d_f>, R"pbdoc(Inspect)pbdoc");
  // m.def("print", &print_mdspan<fast_deconv::core::device_span2d_f>, R"pbdoc(Print)pbdoc");
  // m.def("print", &print_mdspan<fd_core::device_span2d_fs>, R"pbdoc(Print)pbdoc");
  // m.def("print", &print_mdspan<fast_deconv::core::device_span2d_f>, R"pbdoc(Print)pbdoc");
}
}  // namespace fast_deconv::python
