
#include <fast_deconv/core/stream_cache.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/matrix/argmax.cuh>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include <cstdio>

namespace nb = nanobind;
using namespace nb::literals;

// void init_wscms_bindings(nb::module_& m);
// void init_matrix_bindings(nb::module_& m);

std::pair<int, float> bind_argmax(nb::ndarray<float>& data, nb::ndarray<bool>& mask, bool use_abs)
{
  if (data.device_type() != nb::device::cuda::value) { return {-1, 0}; }

  fast_deconv::core::span_2d<float> data_span{data.data(), data.shape(0), data.shape(1)};
  fast_deconv::core::span_2d<bool> mask_span{mask.data(), mask.shape(0), mask.shape(1)};

  auto& resources = fast_deconv::core::cached_stream_resources();
  return fast_deconv::matrix::test_argmax(data_span, mask_span, use_abs, resources);
  // return {0, 0};
}

NB_MODULE(pyfast_deconv, m)
{
  m.doc() = "fast_deconv nanobind module";

  m.def("argmax", &bind_argmax);

  // init_wscms_bindings(m);
  // init_matrix_bindings(m);

  m.def("inspect", [](const nb::ndarray<>& a) {
    printf("Array data pointer : %p\n", a.data());
    printf("Array dimension : %zu\n", a.ndim());
    for (size_t i = 0; i < a.ndim(); ++i) {
      printf("Array dimension [%zu] : %zu\n", i, a.shape(i));
      printf("Array stride    [%zu] : %zd\n", i, a.stride(i));
    }
    printf("Device ID = %u (cpu=%i, cuda=%i)\n",
           a.device_id(),
           int(a.device_type() == nb::device::cpu::value),
           int(a.device_type() == nb::device::cuda::value));
    printf("Array dtype: int16=%i, uint32=%i, float32=%i\n",
           a.dtype() == nb::dtype<int16_t>(),
           a.dtype() == nb::dtype<uint32_t>(),
           a.dtype() == nb::dtype<float>());
  });
}
