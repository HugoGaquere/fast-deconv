#include "fast_deconv/core/stream_resources.hpp"

#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/matrix/argmax.cuh>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

namespace nb = nanobind;
using namespace nb::literals;

std::pair<int, float> bind_argmax(nb::ndarray<float>& data, nb::ndarray<bool>& mask, bool use_abs)
{
  if (data.device_type() != nb::device::cuda::value) { return {-1, 0}; }

  fast_deconv::core::span_2d<float> data_span{data.data(), data.shape(0), data.shape(1)};
  fast_deconv::core::span_2d<bool> mask_span{mask.data(), mask.shape(0), mask.shape(1)};

  fast_deconv::core::stream_resources resources;
  // return fast_deconv::matrix::test_argmax(data_span, mask_span, data.size(), use_abs, resources);
  return {0, 0};
}

void init_argmax_bindings(nb::module_& m) { m.def("argmax", &bind_argmax); }
