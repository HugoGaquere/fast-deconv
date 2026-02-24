#include "fast_deconv_bindings.hpp"

#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/algorithm/wscms.hpp>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace fd_algo_wscms  = fast_deconv::algorithm::wscms;
namespace fd_core        = fast_deconv::core;

namespace fast_deconv::python {

void bind_wscms(py::module_& m)
{
  py::class_<fd_algo_wscms::MinorCycleContext>(m, "MinorCycleContext")
    .def(py::init<
      fd_core::device_span4d<float>,
      fd_core::host_span2d<int>,
      fd_core::device_span2d<float>,
      fd_core::device_vect<float>,
      bool, float, uint, bool>(),
      py::arg("jones_norm"),
      py::arg("map_pixels_facets"),
      py::arg("Xdes"),
      py::arg("sqrt_weights"),
      py::arg("beam_enable"),
      py::arg("peak_factor"),
      py::arg("n_subminor_iter"),
      py::arg("do_abs"));

  py::class_<fd_algo_wscms::ComponentEntry>(m, "ComponentEntry")
    .def_readonly("x", &fd_algo_wscms::ComponentEntry::x)
    .def_readonly("y", &fd_algo_wscms::ComponentEntry::y)
    .def_readonly("scale_idx", &fd_algo_wscms::ComponentEntry::scale_idx)
    .def_readonly("gain", &fd_algo_wscms::ComponentEntry::gain)
    .def_readonly("n_coeffs", &fd_algo_wscms::ComponentEntry::n_coeffs)
    .def("get_coeffs", [](const fd_algo_wscms::ComponentEntry& self) {
      py::list result;
      for (int i = 0; i < self.n_coeffs; i++)
        result.append(self.coeffs[i]);
      return result;
    });

  m.def(
    "minor_cycle",
    [](fd_core::device_span4d<float> dirty,
       fd_core::device_span4d<float> scaled_dirty,
       fd_core::device_span6d<float> psfs,
       fd_core::device_span6d<float> psfs_2,
       fd_core::device_span4d<bool> mask,
       fd_core::host_span2d<float> gains,
       std::uint32_t scale_idx,
       fd_algo_wscms::MinorCycleContext& ctx) -> py::list {
      auto entries = fd_algo_wscms::minor_cycle(
        dirty, scaled_dirty, psfs, psfs_2, mask, gains, scale_idx, ctx);

      // Convert to Python list of (coords, coeffs, scale_idx, gain) tuples
      py::list result;
      for (const auto& e : entries) {
        py::tuple coords = py::make_tuple(e.y, e.x);  // (row, col) order
        py::list coeffs;
        for (int j = 0; j < e.n_coeffs; j++)
          coeffs.append(e.coeffs[j]);
        result.append(py::make_tuple(coords, coeffs, e.scale_idx, e.gain));
      }
      return result;
    },
    py::arg("dirty"),
    py::arg("scaled_dirty"),
    py::arg("psfs"),
    py::arg("psfs_2"),
    py::arg("mask"),
    py::arg("gains"),
    py::arg("scale_idx"),
    py::arg("ctx"),
    R"pbdoc(Run the WSCMS sub-minor loop. Returns a list of (coords, coeffs, scale_idx, gain) tuples.)pbdoc");

}
}  // namespace fast_deconv::python
