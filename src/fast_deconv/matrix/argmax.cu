#include <cub/cub.cuh>
#include <fast_deconv/matrix/argmax.hpp>

namespace fast_deconv::matrix {

void argmax_async(argmax_workspace& ws, const float* d_data)
{
#if CUB_VERSION >= 300000
  CHECK_CUDA(cub::DeviceReduce::ArgMax(ws.d_temp.data_handle(), ws.temp_storage_bytes, d_data,
                                       ws.d_peak_value.data_handle(), ws.d_peak_index.data_handle(), ws.n_elements,
                                       ws.stream_res.cuda_stream));
#else
  CHECK_CUDA(cub::DeviceReduce::ArgMax(ws.d_temp.data_handle(), ws.temp_storage_bytes, d_data, ws.d_peak.data_handle(),
                                       ws.n_elements, ws.stream_res.cuda_stream));
#endif
}

std::tuple<float, int> argmax(argmax_workspace& ws, const float* d_data)
{
  argmax_async(ws, d_data);
#if CUB_VERSION >= 300000
  CHECK_CUDA(cudaMemcpyAsync(&ws.h_peak_value, ws.d_peak_value.data_handle(), sizeof(float), cudaMemcpyDeviceToHost,
                             ws.stream_res.cuda_stream));
  CHECK_CUDA(cudaMemcpyAsync(&ws.h_peak_index, ws.d_peak_index.data_handle(), sizeof(int), cudaMemcpyDeviceToHost,
                             ws.stream_res.cuda_stream));
  ws.stream_res.sync();
  return {ws.h_peak_value, ws.h_peak_index};
#else
  CHECK_CUDA(cudaMemcpyAsync(&ws.h_peak, ws.d_peak.data_handle(), sizeof(cub::KeyValuePair<int, float>),
                             cudaMemcpyDeviceToHost, ws.stream_res.cuda_stream));
  ws.stream_res.sync();
  return {ws.h_peak.value, ws.h_peak.key};
#endif
}

}  // namespace fast_deconv::matrix
