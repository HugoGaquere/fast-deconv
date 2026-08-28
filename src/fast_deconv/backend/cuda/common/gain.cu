#include <cub/cub.cuh>
#include <emu/submdspan.hpp>
#include <fast_deconv/common/gain.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/linalg/linalg.hpp>

namespace fast_deconv::common {

std::vector<float> compute_gain_batched(const core::exec_ctx& ctx, const core::span4d<float>& psfs,
                                        const core::span1d<float>& weights_freq, float gamma)
{
  const int n_batch = psfs.extent(0);
  const int n_ch = psfs.extent(1);
  const int psf_npix = psfs.extent(2) * psfs.extent(3);

  // Scratch buffer for the weighted mean PSF of each psf
  auto wmean = ctx.alloc_mdcontainer_async<float>(psf_npix);
  // Per-facet max values on device (bulk-copied to host after the loop)
  auto d_maxes = ctx.alloc_mdcontainer_async<float>(n_batch);

  // CUB DeviceReduce::Max temp storage (query once, reuse across facets)
  size_t temp_bytes = 0;
  cub::DeviceReduce::Max(nullptr, temp_bytes, wmean.data_handle(), d_maxes.data_handle(), psf_npix, ctx.cuda_stream);
  auto d_temp = ctx.alloc_mdcontainer_async<char>(temp_bytes);

  for (int b = 0; b < n_batch; b++) {
    // 1. Weighted mean over channels
    const float* psf_ptr = psfs.data_handle() + psfs.mapping()(b, 0, 0, 0);
    linalg::weighted_sum_async(ctx, psf_ptr, weights_freq.data_handle(), wmean.data_handle(), n_ch, psf_npix);
    // 2. Max reduction -> d_maxes[f]
    cub::DeviceReduce::Max(d_temp.data_handle(), temp_bytes, wmean.data_handle(), d_maxes.data_handle() + b, psf_npix,
                           ctx.cuda_stream);
  }

  // Bulk copy all max values to host and compute gains
  std::vector<float> gains(n_batch);
  CHECK_CUDA(cudaMemcpyAsync(gains.data(), d_maxes.data_handle(), sizeof(float) * n_batch, cudaMemcpyDeviceToHost,
                             ctx.cuda_stream));
  ctx.wait();

  for (int b = 0; b < n_batch; b++) {
    gains[b] = gamma / gains[b];
  }

  return gains;
}

std::vector<float> compute_all_gains_batched(const core::exec_ctx& ctx, const core::span5d<float>& psfs,
                                             const core::span1d<float>& weights_freq, float gamma)
{
  const int n_scales = psfs.extent(0);
  const int n_facets = psfs.extent(1);
  std::vector<float> all_gains;
  all_gains.reserve(n_scales * n_facets);

  // For scale 0, gains is equal to gamma
  auto gains = std::vector<float>(n_facets, gamma);
  all_gains.insert(all_gains.end(), gains.begin(), gains.end());

  for (int i = 1; i < n_scales; i++) {
    core::span4d<float> current_psf = emu::submdspan(psfs, i);
    auto gains = compute_gain_batched(ctx, current_psf, weights_freq, gamma);
    all_gains.insert(all_gains.end(), gains.begin(), gains.end());
  }
  return all_gains;
}

}  // namespace fast_deconv::common