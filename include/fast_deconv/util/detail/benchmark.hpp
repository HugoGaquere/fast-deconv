#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::util::detail {

inline void flush_l2_cache()
{
  int device  = 0;
  int l2_size = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device));
  size_t flush_size = l2_size * 2;

  float* d_F = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_F, flush_size));
  CHECK_CUDA(cudaMemset((void*)d_F, 0, flush_size));
  CHECK_CUDA(cudaFree(d_F));
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_LAST_CUDA_ERROR();
}

inline float percentile(std::vector<float>& v, float p)
{
  size_t n = v.size();
  std::sort(v.begin(), v.end());
  float f_index = (p / 100.0) * (n - 1);
  size_t lower  = std::floor(f_index);
  size_t upper  = std::ceil(f_index);
  if (lower == upper) return v[lower];
  return v[lower] + (v[upper] - v[lower]) * (f_index - lower);
}

inline float mean(std::vector<float>& v)
{
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

inline float standard_deviation(std::vector<float>& v)
{
  float _mean              = mean(v);
  float sum_sq_differences = 0.0;
  for (auto i : v)
    sum_sq_differences += (i - _mean) * (i - _mean);
  return std::sqrt(sum_sq_differences / v.size());
}

inline void print_bench_results(std::vector<float>& runtimes)
{
  std::sort(runtimes.begin(), runtimes.end());
  float _mean   = mean(runtimes);
  float _std    = standard_deviation(runtimes);
  float _median = percentile(runtimes, 50.0);
  float _min    = *std::min_element(runtimes.begin(), runtimes.end());
  float _p99    = percentile(runtimes, 99.0);

  std::cout << std::endl;
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Benchmark results (" << runtimes.size() << " iterations)" << std::endl;
  std::cout << "===================================" << std::endl;
  std::cout << "Average execution time: " << _mean << "ms" << std::endl;
  std::cout << "Standard deviation: " << _std << std::endl;
  std::cout << "Coefficient of variation: " << (_std / _mean) * 100 << "%" << std::endl;
  std::cout << "Minimum time: " << _min << "ms" << std::endl;
  std::cout << "Median: " << _median << "ms" << std::endl;
  std::cout << "P_99: " << _p99 << "ms" << std::endl;
  std::cout << std::endl;
}
}  // namespace fast_deconv::util::detail
