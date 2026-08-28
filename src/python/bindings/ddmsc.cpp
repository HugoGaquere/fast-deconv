#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/algorithm/ddmsc.hpp>
#include <fast_deconv/algorithm/ddmsc_types.hpp>
#include <fast_deconv/core/memory_types.hpp>

#include "fast_deconv_bindings.hpp"

namespace py = pybind11;
namespace core = fast_deconv::core;
namespace ddmsc = fast_deconv::algorithm::ddmsc;

namespace fast_deconv::python {

void bind_ddmsc(py::module_& m)
{
  auto ddmsc_module = m.def_submodule("ddmsc", "DDMSC module");

  py::native_enum<fast_deconv::common::convergence_status>(ddmsc_module, "ConvergenceStatus", "enum.Enum")
      .value("running", fast_deconv::common::convergence_status::running)
      .value("converged", fast_deconv::common::convergence_status::converged)
      .value("diverged", fast_deconv::common::convergence_status::diverged)
      .value("max_iterations", fast_deconv::common::convergence_status::max_iterations)
      .value("all_scales_stalled", fast_deconv::common::convergence_status::all_scales_stalled)
      .value("no_components", fast_deconv::common::convergence_status::no_components)
      .finalize();

  py::class_<ddmsc::ddmsc_result>(ddmsc_module, "DDMSCResult")
      .def_readonly("peak_coords", &ddmsc::ddmsc_result::peak_coords)
      .def_readonly("scales", &ddmsc::ddmsc_result::scales)
      .def_readonly("gains", &ddmsc::ddmsc_result::gains)
      .def_readonly("coeffs", &ddmsc::ddmsc_result::coeffs)
      .def_readonly("final_flux", &ddmsc::ddmsc_result::final_flux)
      .def_readonly("stop_flux", &ddmsc::ddmsc_result::stop_flux)
      .def_readonly("total_iterations", &ddmsc::ddmsc_result::total_iterations)
      .def_readonly("status", &ddmsc::ddmsc_result::status);

  py::class_<ddmsc::Ddmsc>(ddmsc_module, "DDMSC")
      .def(py::init<const core::host_span4d<float>&, const core::host_span2d<float>&, const core::host_span2d<bool>&,
                    const core::host_span1d<float>&, const core::host_span1d<float>&, const core::host_span2d<int>&,
                    int, int, int, float, int>(),
           py::arg("raw_psfs"), py::arg("xdes"), py::arg("scale_mask"), py::arg("scale_sigmas"), py::arg("scale_bias"),
           py::arg("map_pixel_facet"), py::arg("dirty_nrow"), py::arg("dirty_ncol"), py::arg("n_freq"),
           py::arg("fft_padding"), py::arg("exec_device") = 0,
           // Every array input stays a host view until the first run() stages it, so all six must outlive the object.
           py::keep_alive<1, 2>(),  // raw_psfs
           py::keep_alive<1, 3>(),  // xdes
           py::keep_alive<1, 4>(),  // scale_mask
           py::keep_alive<1, 5>(),  // scale_sigmas
           py::keep_alive<1, 6>(),  // scale_bias
           py::keep_alive<1, 7>())  // map_pixel_facet
      .def("run", &ddmsc::Ddmsc::run, py::arg("dirty"), py::arg("jones_norm"), py::arg("weights_freq"),
           R"pbdoc(Run the DDMSC minor-cycle loop with stall/divergence checks.)pbdoc")
      .def_property("clean_negative", &ddmsc::Ddmsc::clean_negative, &ddmsc::Ddmsc::set_clean_negative)
      .def_property("peak_factor", &ddmsc::Ddmsc::peak_factor, &ddmsc::Ddmsc::set_peak_factor)
      .def_property("gamma", &ddmsc::Ddmsc::gamma, &ddmsc::Ddmsc::set_gamma)
      .def_property("max_sub_iteration", &ddmsc::Ddmsc::max_sub_iteration, &ddmsc::Ddmsc::set_max_sub_iteration)
      .def_property("flux_threshold", &ddmsc::Ddmsc::flux_threshold, &ddmsc::Ddmsc::set_flux_threshold)
      .def_property("stop_rms_factor", &ddmsc::Ddmsc::stop_rms_factor, &ddmsc::Ddmsc::set_stop_rms_factor)
      .def_property("stop_peak_factor", &ddmsc::Ddmsc::stop_peak_factor, &ddmsc::Ddmsc::set_stop_peak_factor)
      .def_property("stop_cycle_factor", &ddmsc::Ddmsc::stop_cycle_factor, &ddmsc::Ddmsc::set_stop_cycle_factor)
      .def_property("stop_sidelobe_level", &ddmsc::Ddmsc::stop_sidelobe_level, &ddmsc::Ddmsc::set_stop_sidelobe_level)
      .def_property("max_iteration", &ddmsc::Ddmsc::max_iteration, &ddmsc::Ddmsc::set_max_iteration)
      .def_property("divergence_factor", &ddmsc::Ddmsc::divergence_factor, &ddmsc::Ddmsc::set_divergence_factor)
      .def_property("stall_threshold", &ddmsc::Ddmsc::stall_threshold, &ddmsc::Ddmsc::set_stall_threshold)
      .def_property("auto_mask", &ddmsc::Ddmsc::auto_mask, &ddmsc::Ddmsc::set_auto_mask)
      .def_property("force_auto_mask", &ddmsc::Ddmsc::force_auto_mask, &ddmsc::Ddmsc::set_force_auto_mask)
      .def_property("auto_mask_peak_threshold", &ddmsc::Ddmsc::auto_mask_peak_threshold,
                    &ddmsc::Ddmsc::set_auto_mask_peak_threshold)
      .def_property("auto_mask_rms_threshold", &ddmsc::Ddmsc::auto_mask_rms_threshold,
                    &ddmsc::Ddmsc::set_auto_mask_rms_threshold);
}
}  // namespace fast_deconv::python
