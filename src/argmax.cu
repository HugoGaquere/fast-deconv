#include "utils.cuh"
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

extern "C" int argmax(float* data, bool* mask, size_t size) {
    using namespace cub;

    // 1. Create a counting iterator [0, 1, 2, ..., size-1]
    CountingInputIterator<int> counting_iter(0);

    // 2. Create the transform iterator that applies masking on-the-fly
    MaskingOp op = {data, mask};
    TransformInputIterator<float, MaskingOp, CountingInputIterator<int>>
        masked_iter(counting_iter, op);

    // 3. Allocate output
    KeyValuePair<int, float>* d_output;
    cudaMalloc(&d_output, sizeof(KeyValuePair<int, float>));

    // 4. Temp storage
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // First call to get temp storage size
    DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, masked_iter, d_output, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // 5. Actual ArgMax call
    DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, masked_iter, d_output, size);

    // 6. Copy result back
    KeyValuePair<int, float> h_output;
    cudaMemcpy(&h_output, d_output, sizeof(KeyValuePair<int, float>), cudaMemcpyDeviceToHost);

    // 7. Free
    cudaFree(d_output);
    cudaFree(d_temp_storage);

    return h_output.key;
}
