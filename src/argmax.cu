#include "argmax.cuh"
#include "utils.cuh"
#include <cmath>
#include <cstdio>
#include <cub/cub.cuh>

struct MaskedArgMaxOp {
    __host__ __device__ MaskedValue operator()(const MaskedValue &a,
                                               const MaskedValue &b) const {
        if (!a.valid)
            return b;
        if (!b.valid)
            return a;
        return (a.value >= b.value) ? a : b;
    }
};


__host__ __device__
MaskedValue::MaskedValue() : index(-1), value(-INFINITY), valid(false) {}

__host__ __device__
MaskedValue::MaskedValue(int i, float v, bool m) : index(i), value(v), valid(m) {}


void ArgmaxContext::init(const size_t size) {
    if (temp_storage)
        return;
    cudaMalloc((void**)&zipped, size * sizeof(MaskedValue));
    cudaMalloc((void**)&output, sizeof(MaskedValue));
    cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, zipped, output, size,
                              MaskedArgMaxOp{}, MaskedValue(),
                              cudaStreamDefault);
    cudaMalloc(&temp_storage, temp_storage_bytes);
}

void ArgmaxContext::free() {
    cudaFree(zipped);
    cudaFree(output);
    cudaFree(temp_storage);
    zipped = nullptr;
    output = nullptr;
    temp_storage = nullptr;
    temp_storage_bytes = 0;
}


__global__ void zip_kernel(const float* data, const bool* mask,
                           MaskedValue* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = MaskedValue{i, data[i], mask[i]};
    }
}

extern "C" int argmax(ArgmaxContext* context, const float* data,
                      const bool* mask, const size_t size) {
    // Zip input
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    zip_kernel<<<blocks, threads>>>(data, mask, context->zipped, size);

    // CHECK_LAST_CUDA_ERROR();


    // Actual reduction
    cub::DeviceReduce::Reduce(context->temp_storage, context->temp_storage_bytes,
                              context->zipped, context->output, size,
                              MaskedArgMaxOp{}, MaskedValue(), cudaStreamDefault);
    // CHECK_LAST_CUDA_ERROR();

    // Copy result
    int h_result;
    cudaMemcpy(&h_result, &(context->output->index), sizeof(int), cudaMemcpyDeviceToHost);

    return h_result;
}
