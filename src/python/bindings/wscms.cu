#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/algorithm/wscms_op.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <pybind11/pybind11.h>

#include "fast_deconv_bindings.hpp"

namespace py = pybind11;

namespace fd_wscms = fast_deconv::algo::wscms;
namespace fd_core = fast_deconv::core;

namespace fast_deconv::python
{
void bind_wscms(py::module_& m)
{
  m.def("subtract_psf_from_dirty_async", &fd_wscms::subtract_psf_from_dirty_async,
        R"pbdoc(Subtract scaled PSF from dirty image)pbdoc");
}
}  // namespace fast_deconv::python
