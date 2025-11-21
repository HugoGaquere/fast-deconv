#include <cuda_runtime.h>

#include <fast_deconv/util/cuda_macros.hpp>
#include <fast_deconv/util/detail/benchmark.hpp>

#include <functional>
#include <vector>

namespace fast_deconv::util {

constexpr int WARMUP_ITERS = 100;
constexpr int ITERS        = 1000;

inline void run_benchmark(std::function<void()> kernel, cudaStream_t stream)
{
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < WARMUP_ITERS; i++) {
    kernel();
    CHECK_LAST_CUDA_ERROR();
  }

  std::vector<float> runtimes(ITERS);
  for (int i = 0; i < ITERS; i++) {
    detail::flush_l2_cache();
    CHECK_CUDA(cudaEventRecord(start, stream));
    kernel();
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    runtimes[i] = milliseconds;
  }
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  detail::print_bench_results(runtimes);
}

}  // namespace fast_deconv::util
