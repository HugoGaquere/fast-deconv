#include <cstdio>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

extern "C" int argmax(float* data, bool* mask, size_t size);

namespace nb = nanobind;


int _argmax(nb::ndarray<float>& data, nb::ndarray<bool>& mask) {
    return argmax(data.data(), mask.data(), data.size());
}

NB_MODULE(kernels, m) {

    m.def("argmax", &_argmax);
    
    m.def("inspect", [](const nb::ndarray<>& a) {
        printf("Array data pointer : %p\n", a.data());
        printf("Array dimension : %zu\n", a.ndim());
        for (size_t i = 0; i < a.ndim(); ++i) {
            printf("Array dimension [%zu] : %zu\n", i, a.shape(i));
            printf("Array stride    [%zu] : %zd\n", i, a.stride(i));
        }
        printf("Device ID = %u (cpu=%i, cuda=%i)\n", a.device_id(),
            int(a.device_type() == nb::device::cpu::value),
            int(a.device_type() == nb::device::cuda::value)
        );
        printf("Array dtype: int16=%i, uint32=%i, float32=%i\n",
            a.dtype() == nb::dtype<int16_t>(),
            a.dtype() == nb::dtype<uint32_t>(),
            a.dtype() == nb::dtype<float>()
        );
    });
}


