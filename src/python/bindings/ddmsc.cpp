#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/algorithm/ddmsc.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <type_traits>
#include <utility>

#include "fast_deconv_bindings.hpp"

namespace py = pybind11;
namespace core = fast_deconv::core;
namespace ddmsc = fast_deconv::algorithm::ddmsc;

namespace fast_deconv::python {

namespace {

// Binds one params field as a read/write property of the session.
template <auto Field>
void bind_param(py::class_<ddmsc::Ddmsc>& cls, const char* name)
{
  using T = std::remove_cvref_t<decltype(std::declval<ddmsc::params&>().*Field)>;
  cls.def_property(
      name, [](const ddmsc::Ddmsc& d) { return d.cfg.*Field; }, [](ddmsc::Ddmsc& d, T v) { d.cfg.*Field = v; });
}

}  // namespace

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
      .def_property_readonly("coeffs",
                             [](const ddmsc::ddmsc_result& r) {
                               const auto n = static_cast<py::ssize_t>(r.peak_coords.size());
                               return py::array_t<float>({n, static_cast<py::ssize_t>(r.n_order)}, r.coeffs.data());
                             })
      .def_property_readonly("n_components", &ddmsc::ddmsc_result::n_components)
      .def_readonly("initial_flux", &ddmsc::ddmsc_result::initial_flux)
      .def_readonly("initial_rms", &ddmsc::ddmsc_result::initial_rms)
      .def_readonly("final_flux", &ddmsc::ddmsc_result::final_flux)
      .def_readonly("final_rms", &ddmsc::ddmsc_result::final_rms)
      .def_readonly("stop_flux", &ddmsc::ddmsc_result::stop_flux)
      .def_readonly("total_iterations", &ddmsc::ddmsc_result::total_iterations)
      .def_readonly("outer_iterations", &ddmsc::ddmsc_result::outer_iterations)
      .def_readonly("status", &ddmsc::ddmsc_result::status);

  py::class_<ddmsc::Ddmsc> cls(ddmsc_module, "DDMSC");
  cls.def(py::init<core::host_span4d<const float>, core::host_span2d<const float>, core::host_span2d<const bool>,
                   core::host_span1d<const float>, core::host_span1d<const float>, core::host_span2d<const int>,
                   std::int64_t, std::int64_t, int>(),
          py::arg("raw_psfs"), py::arg("xdes"), py::arg("mask"), py::arg("scale_sigmas"), py::arg("scale_bias"),
          py::arg("map_pixel_facet"), py::arg("dirty_nrow"), py::arg("dirty_ncol"), py::arg("exec_device") = 0,
          // Borrowed: scale_bias, map_pixel_facet, plus raw_psfs/xdes/mask on the host backend.
          py::keep_alive<1, 2>(),  // raw_psfs
          py::keep_alive<1, 3>(),  // xdes
          py::keep_alive<1, 4>(),  // mask
          py::keep_alive<1, 6>(),  // scale_bias
          py::keep_alive<1, 7>())  // map_pixel_facet
      .def("run", &ddmsc::Ddmsc::run, py::arg("dirty"), py::arg("jones_norm"), py::arg("weights_freq"),
           R"pbdoc(Run the DDMSC minor-cycle loop with stall/divergence checks.)pbdoc");

  // Python names differ for max_sub_iteration, stall_threshold, auto_mask and force_auto_mask.
  bind_param<&ddmsc::params::clean_negative>(cls, "clean_negative");
  bind_param<&ddmsc::params::peak_factor>(cls, "peak_factor");
  bind_param<&ddmsc::params::gamma>(cls, "gamma");
  bind_param<&ddmsc::params::max_clean_iteration>(cls, "max_sub_iteration");
  bind_param<&ddmsc::params::flux_threshold>(cls, "flux_threshold");
  bind_param<&ddmsc::params::stop_rms_factor>(cls, "stop_rms_factor");
  bind_param<&ddmsc::params::stop_peak_factor>(cls, "stop_peak_factor");
  bind_param<&ddmsc::params::stop_cycle_factor>(cls, "stop_cycle_factor");
  bind_param<&ddmsc::params::stop_sidelobe_level>(cls, "stop_sidelobe_level");
  bind_param<&ddmsc::params::max_iteration>(cls, "max_iteration");
  bind_param<&ddmsc::params::divergence_factor>(cls, "divergence_factor");
  bind_param<&ddmsc::params::scale_stall_threshold>(cls, "stall_threshold");
  bind_param<&ddmsc::params::enable_auto_mask>(cls, "auto_mask");
  bind_param<&ddmsc::params::force_enable_auto_mask>(cls, "force_auto_mask");
  bind_param<&ddmsc::params::auto_mask_peak_threshold>(cls, "auto_mask_peak_threshold");
  bind_param<&ddmsc::params::auto_mask_rms_threshold>(cls, "auto_mask_rms_threshold");
}
}  // namespace fast_deconv::python
