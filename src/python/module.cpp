
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

namespace {

fast_deconv::core::stream_resources& python_stream_resources()
{
  static fast_deconv::core::stream_resources resources{};
  return resources;
}

template <typename ArrayT>
bool is_c_contiguous(const ArrayT& array)
{
  const ssize_t ndim = array.ndim();
  if (ndim == 0) { return true; }

  ssize_t stride = array.stride(ndim - 1);
  if (stride <= 0) { return false; }

  for (ssize_t axis = ndim - 2; axis >= 0; --axis) {
    const ssize_t expected = static_cast<ssize_t>(array.shape(axis + 1)) * stride;
    if (array.stride(axis) != expected) { return false; }
    stride = expected;
  }

  return true;
}

template <typename T>
core::span_2d<T> make_span_2d(nb::ndarray<const T>& array)
{
  return {const_cast<T*>(array.data()), array.shape(0), array.shape(1)};
}

}  // namespace

std::pair<int, float> bind_argmax(nb::ndarray<const float>& data,
                                  nb::ndarray<const bool>& mask,
                                  bool use_abs)
{
  if (data.device_type() != nb::device::cuda::value)
    throw nb::value_error("argmax input must live on a CUDA device");
  if (mask.device_type() != nb::device::cuda::value)
    throw nb::value_error("mask input must live on a CUDA device");

  if (data.ndim() != 2 || mask.ndim() != 2)
    throw nb::value_error("argmax expects 2-D inputs");

  if (data.shape(0) != mask.shape(0) || data.shape(1) != mask.shape(1))
    throw nb::value_error("data and mask must share the same shape");

  if (!is_c_contiguous(data) || !is_c_contiguous(mask))
    throw nb::value_error("argmax inputs must be row-major contiguous");

  auto data_span = make_span_2d(data);
  auto mask_span = make_span_2d(mask);

  auto& resources = python_stream_resources();
  return fast_deconv::matrix::test_argmax(data_span, mask_span, use_abs, resources);
  // return {0, 0};
}

void bind_argmax_async(nb::ndarray<const float>& data,
                       nb::ndarray<const bool>& mask,
                       bool use_abs,
                       nb::uintptr out_ptr,
                       nb::uintptr stream_ptr)
{
  if (data.device_type() != nb::device::cuda::value)
    throw nb::value_error("argmax input must live on a CUDA device");
  if (mask.device_type() != nb::device::cuda::value)
    throw nb::value_error("mask input must live on a CUDA device");

  if (data.ndim() != 2 || mask.ndim() != 2)
    throw nb::value_error("argmax expects 2-D inputs");

  if (data.shape(0) != mask.shape(0) || data.shape(1) != mask.shape(1))
    throw nb::value_error("data and mask must share the same shape");

  if (!is_c_contiguous(data) || !is_c_contiguous(mask))
    throw nb::value_error("argmax inputs must be row-major contiguous");

  if (out_ptr.value() == 0)
    throw nb::value_error("device output pointer must be non-null");

  auto& resources = python_stream_resources();
  auto* out = reinterpret_cast<std::pair<int, float>*>(out_ptr.value());

  cudaStream_t stream = stream_ptr.value() == 0
                          ? resources.stream
                          : reinterpret_cast<cudaStream_t>(stream_ptr.value());

  auto data_span = make_span_2d(data);
  auto mask_span = make_span_2d(mask);

  fast_deconv::matrix::argmax_async(out, data_span, mask_span, use_abs, resources, stream);
}

NB_MODULE(pyfast_deconv, m)
{
  m.doc() = "fast_deconv nanobind module";

  m.def("argmax", &bind_argmax);
  m.def("argmax_async", &bind_argmax_async, "out_ptr"_a, "stream"_a = nb::uintptr{0});

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
