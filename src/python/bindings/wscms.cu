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
      .def_readonly("final_flux", &wscms::wscms_result::final_flux);

  py::class_<wscms::Wscms>(wscms_module, "Wscms")
      .def(py::init<const core::device_span6d<float>&, const core::device_span4d<float>&,
                    const core::device_span2d<float>&, const core::device_span2d<bool>&,
                    const core::device_vect<float>&, const core::host_vect<float>&,
                    const core::host_span2d<int>&, const core::host_span2d<float>&, int, int, float,
                    bool, float, int>(),
           py::arg("psfs"), py::arg("psfs_2"), py::arg("xdes"), py::arg("scale_masks"),
           py::arg("scale_sigmas"), py::arg("scale_bias"), py::arg("map_pixel_facet"),
           py::arg("gains"), py::arg("dirty_nrows"), py::arg("dirty_ncols"), py::arg("peak_factor"),
           py::arg("clean_negative"), py::arg("fft_padding"), py::arg("exec_device") = 0,
           py::keep_alive<1, 2>(),   // psfs
           py::keep_alive<1, 3>(),   // psfs_2
           py::keep_alive<1, 4>(),   // xdes
           py::keep_alive<1, 5>(),   // scale_masks
           py::keep_alive<1, 6>(),   // scale_sigmas
           py::keep_alive<1, 7>(),   // scale_bias
           py::keep_alive<1, 8>(),   // map_pixel_facet
           py::keep_alive<1, 9>())   // gains
      .def("run", &wscms::Wscms::run, py::arg("dirty"), py::arg("mean_residual"),
           py::arg("jones_norm"), py::arg("weights_freq"), py::arg("max_iterations"),
           R"pbdoc(Run the WSCMS algorithm)pbdoc")
      .def_property("peak_factor", &wscms::Wscms::peak_factor, &wscms::Wscms::set_peak_factor)
      .def_property("clean_negative", &wscms::Wscms::clean_negative,
                    &wscms::Wscms::set_clean_negative);
}
}  // namespace fast_deconv::python
