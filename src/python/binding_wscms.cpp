#include <cstdio>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include "fast_deconv/wscms.h"

namespace nb = nanobind;
using namespace nb::literals;

void init_wsms_bindings(nb::module_& m) {
    
    //nb::class_<WSCMS>(m, "WSCMS")
    //    .def(nb::init<>())
    //    .def("run", &fast_deconv::wscms::WSCMS::run_subminor_loop);

    // TODO: docstring
}
