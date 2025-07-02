#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char *const file, const int line) {
  cudaError_t const err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    // We don't exit when we encounter CUDA errors in this example.
    // std::exit(EXIT_FAILURE);
  }
}

__global__ void add_one_kernel(float *data, size_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    data[i] += 1.0f;
}

extern "C" void add_one_cuda(float *data, size_t size) {
  float *d_data = nullptr;
  cudaMalloc(&d_data, size * sizeof(float));
  cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  add_one_kernel<<<blocks, threads>>>(d_data, size);

  cudaDeviceSynchronize();
  CHECK_LAST_CUDA_ERROR();

  // Wait for completion
  cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
}
