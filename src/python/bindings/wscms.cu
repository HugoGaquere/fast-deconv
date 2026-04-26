#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/algorithm/wscms_class.hpp>
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

  py::class_<wscms::wscms_result>(wscms_module, "WscmsResult")
      .def_readonly("peak_coords", &wscms::wscms_result::peak_coords)
      .def_readonly("scales", &wscms::wscms_result::scales)
      .def_readonly("gains", &wscms::wscms_result::gains)
      .def_readonly("coeffs", &wscms::wscms_result::coeffs)
      .def_readonly("final_flux", &wscms::wscms_result::final_flux)
      .def_readonly("total_iterations", &wscms::wscms_result::total_iterations);

  py::class_<wscms::Wscms>(wscms_module, "Wscms")
      .def(py::init<const core::device_span4d<float>&, const core::device_span2d<float>&,
                    const core::device_span2d<bool>&, const core::device_vect<float>&, const core::host_vect<float>&,
                    const core::host_span2d<int>&, int, int, int, float, int>(),
           py::arg("raw_psfs"), py::arg("xdes"), py::arg("scale_mask"), py::arg("scale_sigmas"), py::arg("scale_bias"),
           py::arg("map_pixel_facet"), py::arg("dirty_nrow"), py::arg("dirty_ncol"), py::arg("n_freq"),
           py::arg("fft_padding"), py::arg("exec_device") = 0, py::keep_alive<1, 2>(),  // raw_psfs
           py::keep_alive<1, 3>(),                                                      // xdes
           py::keep_alive<1, 4>(),                                                      // scale_mask
           py::keep_alive<1, 5>(),                                                      // scale_sigmas
           py::keep_alive<1, 6>(),                                                      // scale_bias
           py::keep_alive<1, 7>())                                                      // map_pixel_facet
      .def("run", &wscms::Wscms::run, py::arg("dirty"), py::arg("jones_norm"), py::arg("weights_freq"),
           R"pbdoc(Run the WSCMS minor-cycle loop with stall/divergence checks.)pbdoc")
      .def("set_scale_mask", &wscms::Wscms::set_scale_mask, py::arg("scale_mask"), py::keep_alive<1, 2>())
      .def_property("clean_negative", &wscms::Wscms::clean_negative, &wscms::Wscms::set_clean_negative)
      .def_property("peak_factor", &wscms::Wscms::peak_factor, &wscms::Wscms::set_peak_factor)
      .def_property("gamma", &wscms::Wscms::gamma, &wscms::Wscms::set_gamma)
      .def_property("max_sub_iteration", &wscms::Wscms::max_sub_iteration, &wscms::Wscms::set_max_sub_iteration)
      .def_property("stop_flux", &wscms::Wscms::stop_flux, &wscms::Wscms::set_stop_flux)
      .def_property("max_iteration", &wscms::Wscms::max_iteration, &wscms::Wscms::set_max_iteration)
      .def_property("divergence_factor", &wscms::Wscms::divergence_factor, &wscms::Wscms::set_divergence_factor)
      .def_property("stall_threshold", &wscms::Wscms::stall_threshold, &wscms::Wscms::set_stall_threshold);
}
}  // namespace fast_deconv::python
