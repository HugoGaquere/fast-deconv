#include <cstdio>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "argmax.cuh"

// extern "C" void add_one(float* data, size_t size);
extern "C" int argmax(const ArgmaxContext* context, const float* data, const bool* mask, const size_t size);

namespace nb = nanobind;

std::unique_ptr<ArgmaxContext> argmax_context;

void _argmax_init(const size_t size) {
    argmax_context = std::make_unique<ArgmaxContext>();
    argmax_context->init(size);
}

void _argmax_free() {
    argmax_context->free();
}

int _argmax(const nb::ndarray<float>& data, const nb::ndarray<bool>& mask) {
    return argmax(argmax_context.get(), data.data(), mask.data(), data.size());
}

NB_MODULE(kernels, m) {

    m.def("argmax", &_argmax);
    m.def("argmax_init", &_argmax_init);
    m.def("argmax_free", &_argmax_free);
    
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


