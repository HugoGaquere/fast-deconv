#pragma once

#include <cub/cub.cuh>
#include <fast_deconv/core/resources.hpp>

namespace fast_deconv::matrix {

struct argmax_workspace {
  const core::stream_resources& stream_res;
  core::device_cont<char> d_temp;
  size_t temp_storage_bytes = 0;
#if CUB_VERSION >= 300000
  core::device_cont<float> d_peak_value;
  core::device_cont<int> d_peak_index;
  float h_peak_value = 0.0f;
  int h_peak_index = 0;
#else
  core::device_cont<cub::KeyValuePair<int, float> > d_peak;
  cub::KeyValuePair<int, float> h_peak{};
#endif
  size_t n_elements = 0;

  argmax_workspace(const core::stream_resources& stream_res, size_t n_elements)
      : stream_res(stream_res), n_elements(n_elements)
  {
#if CUB_VERSION >= 300000
    d_peak_value = stream_res.alloc_mdcontainer_async<float>(1);
    d_peak_index = stream_res.alloc_mdcontainer_async<int>(1);

    CHECK_CUDA(cub::DeviceReduce::ArgMax(nullptr, temp_storage_bytes, static_cast<const float*>(nullptr),
                                         d_peak_value.data_handle(), d_peak_index.data_handle(), n_elements,
                                         stream_res.cuda_stream));
#else
    d_peak = stream_res.alloc_mdcontainer_async<cub::KeyValuePair<int, float> >(1);

    CHECK_CUDA(cub::DeviceReduce::ArgMax(nullptr, temp_storage_bytes, static_cast<const float*>(nullptr),
                                         d_peak.data_handle(), n_elements, stream_res.cuda_stream));
#endif

    d_temp = stream_res.alloc_mdcontainer_async<char>(temp_storage_bytes);
  }

  argmax_workspace(const argmax_workspace&) = delete;
  argmax_workspace& operator=(const argmax_workspace&) = delete;
};

void argmax_async(argmax_workspace& ws, const float* d_data);
std::tuple<float, int> argmax(argmax_workspace& ws, const float* d_data);

}  // namespace fast_deconv::matrix
