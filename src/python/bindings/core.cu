#include <fast_deconv/core/stream_resources.hpp>
#include <pybind11/pybind11.h>

#include "fast_deconv_bindings.hpp"

#include <cstdint>
#include <memory>

namespace py = pybind11;

namespace fd_core = fast_deconv::core;

namespace fast_deconv::python
{
void bind_core(py::module_& m)
{
  py::class_<fd_core::stream_resources>(m, "stream_resources")
      .def(py::init<>(), "Create stream_resources with a new internal CUDA stream.")
      .def_static(
          "from_cupy_stream",
          [](py::object cupy_stream) {
            // CuPy streams expose .ptr as the raw cudaStream_t pointer
            auto ptr = cupy_stream.attr("ptr").cast<std::uintptr_t>();
            return std::make_unique<fd_core::stream_resources>(reinterpret_cast<cudaStream_t>(ptr));
          },
          py::arg("stream"),
          "Create stream_resources from a CuPy stream. "
          "The CuPy stream must outlive the stream_resources object.");
}
}  // namespace fast_deconv::python
