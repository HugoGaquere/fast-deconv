#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/algorithm/wscms.hpp>
#include <fast_deconv/algorithm/wscms_op.hpp>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/algorithm/clean_dirties_op.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fast_deconv_bindings.hpp"

namespace py = pybind11;

namespace fd_wscms = fast_deconv::algo::wscms;
namespace fd_algo  = fast_deconv::algorithm::wscms;
namespace fd_core  = fast_deconv::core;

namespace fast_deconv::python
{

static fd_algo::MinorCycleContext make_minor_cycle_context(
    fd_core::span_6d<float> psfs,
    fd_core::span_6d<float> psfs_2,
    fd_core::span_4d<float> jones_norm,
    fd_core::span_2d<float> gains,
    fd_core::span_2d<bool> mask,
    fd_core::mdspan<int, 1> map_pixels_facets,
    fd_core::span_2d<float> Xdes,
    fd_core::span_1d<float> sqrt_weights,
    bool beam_enable,
    float peak_factor,
    int n_subminor_iter,
    bool do_abs)
{
  return fd_algo::MinorCycleContext{
      psfs, psfs_2, jones_norm, gains, mask, map_pixels_facets,
      Xdes, sqrt_weights, beam_enable, peak_factor, n_subminor_iter, do_abs};
}

void bind_wscms(py::module_& m)
{
  // Helper to create MinorCycleContext - split into a named function to avoid
  // template depth issues with NVCC + pybind11 lambdas with many parameters.
  py::class_<fd_algo::MinorCycleContext>(m, "MinorCycleContext");

  m.def("make_minor_cycle_context", &make_minor_cycle_context,
    py::arg("psfs"), py::arg("psfs_2"),
    py::arg("jones_norm"), py::arg("gains"), py::arg("mask"),
    py::arg("map_pixels_facets"),
    py::arg("Xdes"), py::arg("sqrt_weights"),
    py::arg("beam_enable"),
    py::arg("peak_factor"), py::arg("n_subminor_iter"), py::arg("do_abs"));

  py::class_<fd_algo::ComponentEntry>(m, "ComponentEntry")
    .def_readonly("x", &fd_algo::ComponentEntry::x)
    .def_readonly("y", &fd_algo::ComponentEntry::y)
    .def_readonly("scale_idx", &fd_algo::ComponentEntry::scale_idx)
    .def_readonly("gain", &fd_algo::ComponentEntry::gain)
    .def_readonly("n_coeffs", &fd_algo::ComponentEntry::n_coeffs)
    .def("get_coeffs", [](const fd_algo::ComponentEntry& self) {
      py::list result;
      for (int i = 0; i < self.n_coeffs; i++)
        result.append(self.coeffs[i]);
      return result;
    });

  m.def("wscms_minor_cycle",
    [](fd_core::span_4d<float> dirty,
       fd_core::span_4d<float> scaled_dirty,
       std::uint32_t scale_idx,
       const fd_algo::MinorCycleContext& ctx,
       fd_core::stream_resources& resources) -> py::list
    {
      // Allocate component buffer on host
      std::vector<fd_algo::ComponentEntry> entries(ctx.n_subminor_iter);
      fd_algo::ComponentBuffer buf{entries.data(), 0, ctx.n_subminor_iter};

      fd_algo::wscms_minor_cycle(dirty, scaled_dirty, scale_idx, ctx, buf, resources);

      // Convert to Python list
      py::list result;
      for (int i = 0; i < buf.count; i++) {
        const auto& e = entries[i];
        py::tuple coords = py::make_tuple(e.x, e.y);
        py::list coeffs;
        for (int j = 0; j < e.n_coeffs; j++)
          coeffs.append(e.coeffs[j]);
        result.append(py::make_tuple(coords, coeffs, e.scale_idx, e.gain));
      }
      return result;
    },
    py::arg("dirty"), py::arg("scaled_dirty"), py::arg("scale_idx"),
    py::arg("ctx"), py::arg("resources"),
    R"pbdoc(
Run the WSCMS sub-minor loop.

Returns a list of (coords, coeffs, scale_idx, gain) tuples.
)pbdoc");

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
