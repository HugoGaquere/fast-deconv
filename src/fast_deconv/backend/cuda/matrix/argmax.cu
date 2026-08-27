#include <cassert>
#include <cub/device/device_reduce.cuh>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::matrix {

argmax_ctx::argmax_ctx(const core::exec_ctx& ctx, std::size_t n_elements) : ctx_(ctx), n_elements_(n_elements)
{
#if CUB_VERSION >= 300000
  d_peak_value_ = ctx_.alloc_ptr_async<float>(1);
  d_peak_index_ = ctx_.alloc_ptr_async<std::int64_t>(1);

  CHECK_CUDA(cub::DeviceReduce::ArgMax(nullptr, temp_bytes_, static_cast<const float*>(nullptr), d_peak_value_.get(),
                                       d_peak_index_.get(), n_elements_, ctx_.cuda_stream));
#else
  d_peak_ = ctx_.alloc_ptr_async<cub::KeyValuePair<int, float> >(1);

  CHECK_CUDA(cub::DeviceReduce::ArgMax(nullptr, temp_bytes_, static_cast<const float*>(nullptr), d_peak_.get(),
                                       n_elements_, ctx_.cuda_stream));
#endif

  d_temp_ = ctx_.alloc_ptr_async<std::byte>(temp_bytes_);
}

void argmax_ctx::run_async(core::span2d<float> data)
{
  assert(data.is_exhaustive());
  assert(data.size() == n_elements_);

#if CUB_VERSION >= 300000
  CHECK_CUDA(cub::DeviceReduce::ArgMax(d_temp_.get(), temp_bytes_, data.data_handle(), d_peak_value_.get(),
                                       d_peak_index_.get(), n_elements_, ctx_.cuda_stream));
#else
  CHECK_CUDA(cub::DeviceReduce::ArgMax(d_temp_.get(), temp_bytes_, data.data_handle(), d_peak_.get(), n_elements_,
                                       ctx_.cuda_stream));
#endif
}

peak argmax_ctx::run(core::span2d<float> data)
{
  run_async(data);

#if CUB_VERSION >= 300000
  CHECK_CUDA(
      cudaMemcpyAsync(&h_peak_value_, d_peak_value_.get(), sizeof(float), cudaMemcpyDeviceToHost, ctx_.cuda_stream));
  CHECK_CUDA(cudaMemcpyAsync(&h_peak_index_, d_peak_index_.get(), sizeof(std::int64_t), cudaMemcpyDeviceToHost,
                             ctx_.cuda_stream));
  ctx_.wait();
  return {h_peak_value_, h_peak_index_};
#else
  CHECK_CUDA(cudaMemcpyAsync(&h_peak_, d_peak_.get(), sizeof(cub::KeyValuePair<int, float>), cudaMemcpyDeviceToHost,
                             ctx_.cuda_stream));
  ctx_.wait();
  return {h_peak_.value, h_peak_.key};
#endif
}

}  // namespace fast_deconv::matrix
