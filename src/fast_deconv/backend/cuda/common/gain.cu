#include <cstdint>
#include <cub/cub.cuh>
#include <emu/submdspan.hpp>
#include <fast_deconv/common/gain.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/linalg/linalg.hpp>

namespace fast_deconv::common {

float compute_gain(const core::exec_ctx& ctx, const core::span3d<const float>& psf,
                   const core::span1d<const float>& weights_freq, float gamma)
{
  // Scratch buffer for the weighted mean PSF
  auto wmean = ctx.alloc_mdcontainer_async<float>(psf.extent(1), psf.extent(2));
  const std::int64_t psf_npix = wmean.mapping().required_span_size();
  auto d_max = ctx.alloc_ptr_async<float>(1);

  // CUB DeviceReduce::Max temp storage
  size_t temp_bytes = 0;
  cub::DeviceReduce::Max(nullptr, temp_bytes, wmean.data_handle(), d_max.get(), psf_npix, ctx.cuda_stream);
  auto d_temp = ctx.alloc_mdcontainer_async<char>(temp_bytes);

  // 1. Weighted mean over channels
  linalg::weighted_sum_async(ctx, psf, weights_freq, wmean);
  // 2. Max reduction
  cub::DeviceReduce::Max(d_temp.data_handle(), temp_bytes, wmean.data_handle(), d_max.get(), psf_npix, ctx.cuda_stream);

  float gain;
  CHECK_CUDA(cudaMemcpyAsync(&gain, d_max.get(), sizeof(float), cudaMemcpyDeviceToHost, ctx.cuda_stream));
  ctx.wait();

  return gamma / gain;
}

std::vector<float> compute_gain_batched(const core::exec_ctx& ctx, const core::span4d<const float>& psfs,
                                        const core::span1d<const float>& weights_freq, float gamma)
{
  const std::int64_t n_batch = psfs.extent(0);

  // Scratch buffer for the weighted mean PSF of each psf
  auto wmean = ctx.alloc_mdcontainer_async<float>(psfs.extent(2), psfs.extent(3));
  const std::int64_t psf_npix = wmean.mapping().required_span_size();
  // Per-facet max values on device (bulk-copied to host after the loop)
  auto d_maxes = ctx.alloc_mdcontainer_async<float>(n_batch);

  // CUB DeviceReduce::Max temp storage (query once, reuse across facets)
  size_t temp_bytes = 0;
  cub::DeviceReduce::Max(nullptr, temp_bytes, wmean.data_handle(), d_maxes.data_handle(), psf_npix, ctx.cuda_stream);
  auto d_temp = ctx.alloc_mdcontainer_async<char>(temp_bytes);

  for (std::int64_t b = 0; b < n_batch; b++) {
    // 1. Weighted mean over channels
    const core::span3d<const float> psf = emu::submdspan(psfs, b);
    linalg::weighted_sum_async(ctx, psf, weights_freq, wmean);
    // 2. Max reduction -> d_maxes[f]
    cub::DeviceReduce::Max(d_temp.data_handle(), temp_bytes, wmean.data_handle(), d_maxes.data_handle() + b, psf_npix,
                           ctx.cuda_stream);
  }

  // Bulk copy all max values to host and compute gains
  std::vector<float> gains(core::narrow<std::size_t>(n_batch));
  ctx.copy(gains.data(), d_maxes.data_handle(), n_batch);
  ctx.wait();

  for (float& g : gains) g = gamma / g;

  return gains;
}

}  // namespace fast_deconv::common
