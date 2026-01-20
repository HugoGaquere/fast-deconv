#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/algorithm/wscms_op.hpp>
#include <fast_deconv/algorithm/clean_dirties_op.hpp>
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

  m.def("clean_dirties_async", &fd_wscms::clean_dirties_async,
        R"pbdoc(
Fused kernel for dirty and scaled_dirty subtraction.

Performs:
  dirty -= psf * coeffs * gain
  scaled_dirty -= psf_2 * gain * mask

Args:
    psf: Convolved PSF for dirty subtraction (nch, npol, h, w)
    psf_2: Convolved PSF for scaled_dirty subtraction (nch, npol, h, w)
    dirty: Dirty image (in-place) (nch, npol, h, w)
    scaled_dirty: Scaled dirty image (in-place) (nch, npol, h, w)
    coeffs: Per-channel coefficients (nch,)
    mask: Mask as float (0.0/1.0) (nch, npol, h, w)
    gain: Gain for dirty subtraction
    resources: Stream resources
)pbdoc");
}
}  // namespace fast_deconv::python
