#include <pybind11/pybind11.h>


//#include <fast_deconv.hpp>


std::string hello_from_bin() { return "Hello from scikit-ext!"; }


/// Start of emu span binding code

#include <emu/pybind11/cast/mdspan.hpp>
#include <vector>

using span_3d = emu::mdspan_3d<double>;

void take_mdspan(const span_3d &m)
{
    fmt::print("Received mdspan with extents: {}\n", emu::extent(m));
}

// Data must outlive the returned mdspan.
std::vector<double> data(4 * 5 * 6, 1.0);

span_3d return_mdspan()
{
    span_3d c(data.data(), 4, 5, 6);
    return c;
}

/// End of emu span binding code

namespace py = pybind11;

PYBIND11_MODULE(_fast_deconv, m)
{
    m.doc() = "fast_deconv hello module";

    m.def("hello_from_bin", &hello_from_bin, R"pbdoc(
        A function that returns a Hello string.
    )pbdoc");

    // m.def("add", &fast_deconv::add, R"pbdoc(
    //     A function that adds two integers.
    // )pbdoc");

    m.def("take_mdspan", &take_mdspan, R"pbdoc(
        A function that takes a 3D mdspan.
    )pbdoc");

    m.def("return_mdspan", &return_mdspan, R"pbdoc(
        A function that makes and returns a 3D mdspan.
    )pbdoc");
}
