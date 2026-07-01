#pragma once

#include <cub/cub.cuh>
#include <fast_deconv/core/resources.hpp>

namespace fast_deconv::matrix {

struct argmax_workspace {
  const core::stream_resources& stream_res;
  char* d_temp = nullptr;
  size_t temp_storage_bytes = 0;
#if CUB_VERSION >= 300000
  float* d_peak_value = nullptr;
  int* d_peak_index = nullptr;
  float h_peak_value = 0.0f;
  int h_peak_index = 0;
#else
  cub::KeyValuePair<int, float>* d_peak = nullptr;
  cub::KeyValuePair<int, float> h_peak{};
#endif
  size_t n_elements = 0;

  argmax_workspace(const core::stream_resources& stream_res, size_t n_elements)
      : stream_res(stream_res), n_elements(n_elements)
  {
#if CUB_VERSION >= 300000
    d_peak_value = stream_res.alloc_async<float>(1);
    d_peak_index = stream_res.alloc_async<int>(1);

    CHECK_CUDA(cub::DeviceReduce::ArgMax(nullptr, temp_storage_bytes, static_cast<const float*>(nullptr), d_peak_value,
                                         d_peak_index, n_elements, stream_res.cuda_stream));
#else
    d_peak = stream_res.alloc_async<cub::KeyValuePair<int, float>>(1);

    CHECK_CUDA(cub::DeviceReduce::ArgMax(nullptr, temp_storage_bytes, static_cast<const float*>(nullptr), d_peak,
                                         n_elements, stream_res.cuda_stream));
#endif

    d_temp = stream_res.alloc_async<char>(temp_storage_bytes);
  }
  ~argmax_workspace()
  {
    if (d_temp) stream_res.free_async(d_temp);
#if CUB_VERSION >= 300000
    if (d_peak_index) stream_res.free_async(d_peak_index);
    if (d_peak_value) stream_res.free_async(d_peak_value);
#else
    if (d_peak) stream_res.free_async(d_peak);
#endif
  }

  argmax_workspace(const argmax_workspace&) = delete;
  argmax_workspace& operator=(const argmax_workspace&) = delete;
};

void argmax_async(argmax_workspace& ws, const float* d_data);
std::tuple<float, int> argmax(argmax_workspace& ws, const float* d_data);

}  // namespace fast_deconv::matrix
