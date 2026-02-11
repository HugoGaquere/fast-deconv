#include "fast_deconv_bindings.hpp"

#include <vector>

#include <emu/pybind11/cast/mdspan.hpp>
#include <fast_deconv/algorithm/wscms.hpp>
#include <fast_deconv/algorithm/wscms_op.hpp>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace fd_wscms = fast_deconv::algo::wscms;
namespace fd_algo  = fast_deconv::algorithm::wscms;
namespace fd_core  = fast_deconv::core;

namespace fast_deconv::python {

// Wrapper that owns the host copy of map_pixels_facets
struct PythonMinorCycleContext {
  std::vector<int> host_map_data;
  fd_algo::MinorCycleContext ctx;
};

static PythonMinorCycleContext make_minor_cycle_context(
  fd_core::device_span4d<float> jones_norm,
  fd_core::device_span2d<int> map_pixels_facets_dev,
  fd_core::device_span2d<float> Xdes,
  fd_core::device_vect<float> sqrt_weights,
  bool beam_enable,
  float peak_factor,
  int n_subminor_iter,
  bool do_abs)
{
  PythonMinorCycleContext result;

  // Copy map_pixels_facets from device to host
  const auto rows = map_pixels_facets_dev.extent(0);
  const auto cols = map_pixels_facets_dev.extent(1);
  result.host_map_data.resize(rows * cols);
  cudaMemcpy(result.host_map_data.data(),
             map_pixels_facets_dev.data_handle(),
             rows * cols * sizeof(int),
             cudaMemcpyDeviceToHost);

  result.ctx = fd_algo::MinorCycleContext{
    jones_norm,
    fd_core::host_span2d<int>(result.host_map_data.data(), rows, cols),
    Xdes,
    sqrt_weights,
    beam_enable,
    peak_factor,
    static_cast<uint>(n_subminor_iter),
    do_abs
  };

  return result;
}

void bind_wscms(py::module_& m)
{
  py::class_<PythonMinorCycleContext>(m, "MinorCycleContext");

  m.def("make_minor_cycle_context",
        &make_minor_cycle_context,
        py::arg("jones_norm"),
        py::arg("map_pixels_facets"),
        py::arg("Xdes"),
        py::arg("sqrt_weights"),
        py::arg("beam_enable"),
        py::arg("peak_factor"),
        py::arg("n_subminor_iter"),
        py::arg("do_abs"));

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

  m.def(
    "wscms_minor_cycle",
    [](fd_core::device_span4d<float> dirty,
       fd_core::device_span4d<float> scaled_dirty,
       fd_core::device_span6d<float> psfs,
       fd_core::device_span6d<float> psfs_2,
       fd_core::device_span4d<bool> mask,
       fd_core::device_span2d<float> gains_dev,
       std::uint32_t scale_idx,
       PythonMinorCycleContext& py_ctx) -> py::list {
      // Copy gains from device to host
      const auto g_rows = gains_dev.extent(0);
      const auto g_cols = gains_dev.extent(1);
      std::vector<float> host_gains(g_rows * g_cols);
      cudaMemcpy(host_gains.data(), gains_dev.data_handle(),
                 g_rows * g_cols * sizeof(float), cudaMemcpyDeviceToHost);
      fd_core::host_span2d<float> gains(host_gains.data(), g_rows, g_cols);

      auto entries = fd_algo::wscms_minor_cycle(
        dirty, scaled_dirty, psfs, psfs_2, mask, gains, scale_idx, py_ctx.ctx);

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
    R"pbdoc(
Run the WSCMS sub-minor loop.

Returns a list of (coords, coeffs, scale_idx, gain) tuples.
)pbdoc");

}
}  // namespace fast_deconv::python
