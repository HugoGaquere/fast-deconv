#include <cub/cub.cuh>
#include <fast_deconv/matrix/argmax.hpp>

namespace fast_deconv::matrix {

void argmax_async(argmax_workspace& ws, const float* d_data)
{
  CHECK_CUDA(cub::DeviceReduce::ArgMax(ws.d_temp, ws.temp_storage_bytes, d_data, ws.d_peak_value, ws.d_peak_index,
                                       ws.n_elements, ws.stream_res.cuda_stream));
}

std::tuple<float, int> argmax(argmax_workspace& ws, const float* d_data)
{
  argmax_async(ws, d_data);
  CHECK_CUDA(cudaMemcpyAsync(&ws.h_peak_value, ws.d_peak_value, sizeof(float), cudaMemcpyDeviceToHost,
                             ws.stream_res.cuda_stream));
  CHECK_CUDA(cudaMemcpyAsync(&ws.h_peak_index, ws.d_peak_index, sizeof(int), cudaMemcpyDeviceToHost,
                             ws.stream_res.cuda_stream));
  ws.stream_res.sync();
  return {ws.h_peak_value, ws.h_peak_index};
}

}  // namespace fast_deconv::matrix
