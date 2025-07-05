#include "utils.cuh"
#include <utility>
#include <cmath>
#include <cstdio>
#include <cub/cub.cuh>

struct MaskingOp {
    const float* data;
    const bool* mask;

    __device__ float operator()(const int& i) const {
        return mask[i] ? data[i] : -INFINITY;
    }
};

struct MaskingOpAbs {
    const float* data;
    const bool* mask;

    __device__ float operator()(const int& i) const {
        return mask[i] ? abs(data[i]) : -INFINITY;
    }
};

template <typename MaskingOpT>
std::pair<int, float> _argmax(float* data, bool* mask, size_t size) {
    using namespace cub;

    // Create the transform iterator that applies masking on-the-fly
    CountingInputIterator<int> counting_iter(0);
    MaskingOpT op = {data, mask};
    TransformInputIterator<float, MaskingOpT, CountingInputIterator<int>> masked_iter(counting_iter, op);

    // Allocate output
    KeyValuePair<int, float>* d_output;
    cudaMalloc(&d_output, sizeof(KeyValuePair<int, float>));

    // Temp storage
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // First call to get temp storage size
    DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, masked_iter, d_output, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Actual ArgMax call
    DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, masked_iter, d_output, size);

    // Copy result back
    KeyValuePair<int, float> h_output;
    cudaMemcpy(&h_output, d_output, sizeof(KeyValuePair<int, float>), cudaMemcpyDeviceToHost);

    // 7. Free
    cudaFree(d_output);
    cudaFree(d_temp_storage);

    return {h_output.key, h_output.value};
}

extern "C" std::pair<int, float> argmax(float* data, bool* mask, size_t size, bool use_abs) {
    if (use_abs) return _argmax<MaskingOpAbs>(data, mask, size);
    return _argmax<MaskingOp>(data, mask, size);
}
