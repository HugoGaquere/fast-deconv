#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/algorithm/wscms.hpp>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>

#include "fast_deconv/core/resources.hpp"
#include "fast_deconv_bindings.hpp"

namespace py = pybind11;
namespace core = fast_deconv::core;
namespace wscms = fast_deconv::algorithm::wscms;

namespace fast_deconv::python {

void bind_wscms(py::module_& m)
{
  auto wscms_module = m.def_submodule("wscms", "WSCMS module");

  py::class_<wscms::WSCMS_params>(wscms_module, "WscmsParams")
      .def(py::init<bool, bool, bool, float, int, int, float>(), py::arg("beam_enable"),
           py::arg("do_abs"), py::arg("per_scale_mask"), py::arg("peak_factor"),
           py::arg("max_subminor_iter"), py::arg("n_scales"), py::arg("padding"))
      .def_readwrite("beam_enable", &wscms::WSCMS_params::beam_enable)
      .def_readwrite("do_abs", &wscms::WSCMS_params::do_abs)
      .def_readwrite("per_scale_mask", &wscms::WSCMS_params::per_scale_mask)
      .def_readwrite("peak_factor", &wscms::WSCMS_params::peak_factor)
      .def_readwrite("max_subminor_iter", &wscms::WSCMS_params::max_subminor_iter)
      .def_readwrite("n_scales", &wscms::WSCMS_params::n_scales)
      .def_readwrite("padding", &wscms::WSCMS_params::padding);

  py::class_<wscms::WSCMS_ctx>(wscms_module, "WscmsCtx")
      .def(py::init<core::device_span4d<float>, core::device_span2d<float>,
                    core::device_vect<float>, core::device_span2d<bool>, core::device_vect<float>,
                    core::host_vect<float>, core::host_span2d<int>, core::host_span2d<float>>(),
           py::arg("jones_norm"), py::arg("xdes"), py::arg("weights_freq"), py::arg("scale_masks"),
           py::arg("scale_sigmas"), py::arg("scale_bias"), py::arg("map_pixel_facet"),
           py::arg("gains"));

  py::class_<wscms::sky_component>(wscms_module, "SkyComponent")
      .def_readonly("row", &wscms::sky_component::row)
      .def_readonly("col", &wscms::sky_component::col)
      .def_readonly("scale_idx", &wscms::sky_component::scale_idx)
      .def_readonly("gain", &wscms::sky_component::gain)
      .def_readonly("coeffs", &wscms::sky_component::coeffs);

  wscms_module.def("run_wscms", &wscms::run_wscms, py::arg("dirty"), py::arg("mean_residual"),
                   py::arg("psfs"), py::arg("psfs_2"), py::arg("ctx"), py::arg("params"),
                   R"pbdoc( Run the full WSCMS algorithm )pbdoc");
}
}  // namespace fast_deconv::python
