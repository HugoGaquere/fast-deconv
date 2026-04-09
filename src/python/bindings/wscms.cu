#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/algorithm/wscms.hpp>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>

#include "fast_deconv_bindings.hpp"

namespace py = pybind11;
namespace core = fast_deconv::core;
namespace wscms = fast_deconv::algorithm::wscms;

namespace fast_deconv::python {

void bind_wscms(py::module_& m)
{
  auto wscms_module = m.def_submodule("wscms", "WSCMS module");

  py::class_<wscms::sky_component>(wscms_module, "SkyComponent")
      .def_readonly("row", &wscms::sky_component::row)
      .def_readonly("col", &wscms::sky_component::col)
      .def_readonly("scale_idx", &wscms::sky_component::scale_idx)
      .def_readonly("gain", &wscms::sky_component::gain)
      .def_readonly("coeffs", &wscms::sky_component::coeffs);

  py::class_<wscms::wscms_result>(wscms_module, "WscmsResult")
      .def_readonly("components", &wscms::wscms_result::components)
      .def_readonly("final_flux", &wscms::wscms_result::final_flux)
      .def_readonly("total_iterations", &wscms::wscms_result::total_iterations);

  py::class_<wscms::Wscms>(wscms_module, "Wscms")
      .def(py::init<const core::device_span5d<float>&, const core::device_span2d<float>&,
                    const core::device_span2d<bool>&, const core::device_vect<float>&,
                    const core::host_vect<float>&, const core::host_span2d<int>&,
                    float, int, int, float, bool, float, int>(),
           py::arg("raw_psfs"), py::arg("xdes"), py::arg("scale_masks"),
           py::arg("scale_sigmas"), py::arg("scale_bias"), py::arg("map_pixel_facet"),
           py::arg("gamma"), py::arg("dirty_nrows"), py::arg("dirty_ncols"), py::arg("peak_factor"),
           py::arg("clean_negative"), py::arg("fft_padding"), py::arg("exec_device") = 0,
           py::keep_alive<1, 2>(),   // raw_psfs
           py::keep_alive<1, 3>(),   // xdes
           py::keep_alive<1, 4>(),   // scale_masks
           py::keep_alive<1, 5>(),   // scale_sigmas
           py::keep_alive<1, 6>(),   // scale_bias
           py::keep_alive<1, 7>())   // map_pixel_facet
      .def("run", &wscms::Wscms::run, py::arg("dirty"), py::arg("jones_norm"),
           py::arg("weights_freq"), py::arg("mask"), py::arg("stop_flux"),
           py::arg("max_iteration"), py::arg("max_sub_iteration"),
           py::arg("divergence_factor"), py::arg("stall_threshold"),
           py::arg("forbidden_scales"),
           R"pbdoc(Run the WSCMS minor-cycle loop with stall/divergence checks.)pbdoc")
      .def_property("peak_factor", &wscms::Wscms::peak_factor, &wscms::Wscms::set_peak_factor)
      .def_property("clean_negative", &wscms::Wscms::clean_negative,
                    &wscms::Wscms::set_clean_negative);
}
}  // namespace fast_deconv::python
