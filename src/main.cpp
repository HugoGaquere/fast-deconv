#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern "C" void add_one_cuda(float* data, size_t size);

namespace py = pybind11;

void add_one(py::array_t<float> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1)
        throw std::runtime_error("input must be 1-D");
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.shape[0];
    add_one_cuda(ptr, size);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "pybind11 CUDA example";
    m.def("add_one", &add_one, "Add one to each element of a float array");
}
