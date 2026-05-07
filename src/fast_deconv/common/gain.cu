#include <cub/cub.cuh>
#include <fast_deconv/common/gain.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/linalg/linalg.hpp>

namespace fast_deconv::common {

std::vector<float> compute_gain_batched(const core::resources& resources, const core::stream_resources& stream_res,
                                        const core::device_span4d<float>& psfs,
                                        const core::device_vect<float>& weights_freq, float gamma)
{
  const int n_batch = psfs.extent(0);
  const int n_ch = psfs.extent(1);
  const int psf_npix = psfs.extent(2) * psfs.extent(3);

  // Scratch buffer for the weighted mean PSF of each psf
  float* wmean = resources.alloc_async<float>(psf_npix, stream_res);
  // Per-facet max values on device (bulk-copied to host after the loop)
  float* d_maxes = resources.alloc_async<float>(n_batch, stream_res);

  // CUB DeviceReduce::Max temp storage (query once, reuse across facets)
  size_t temp_bytes = 0;
  cub::DeviceReduce::Max(nullptr, temp_bytes, wmean, d_maxes, psf_npix, stream_res.cuda_stream);
  char* d_temp = resources.alloc_async<char>(temp_bytes, stream_res);

  for (int b = 0; b < n_batch; b++) {
    // 1. Weighted mean over channels
    const float* psf_ptr = psfs.data_handle() + psfs.mapping()(b, 0, 0, 0);
    linalg::weighted_sum_async(stream_res, psf_ptr, weights_freq.data_handle(), wmean, n_ch, psf_npix);
    // 2. Max reduction -> d_maxes[f]
    cub::DeviceReduce::Max(d_temp, temp_bytes, wmean, d_maxes + b, psf_npix, stream_res.cuda_stream);
  }

  // Bulk copy all max values to host and compute gains
  std::vector<float> gains(n_batch);
  CHECK_CUDA(
      cudaMemcpyAsync(gains.data(), d_maxes, sizeof(float) * n_batch, cudaMemcpyDeviceToHost, stream_res.cuda_stream));
  stream_res.sync();
  resources.free_async(d_temp, stream_res);
  resources.free_async(d_maxes, stream_res);
  resources.free_async(wmean, stream_res);

  for (int b = 0; b < n_batch; b++) {
    gains[b] = gamma / gains[b];
  }

  return gains;
}

std::vector<float> compute_all_gains_batched(const core::resources& resources, const core::stream_resources& stream_res,
                                             const core::device_span5d<float>& psfs,
                                             const core::device_vect<float>& weights_freq, float gamma)
{
  const int n_scales = psfs.extent(0);
  const int n_facets = psfs.extent(1);
  std::vector<float> all_gains;
  all_gains.reserve(n_scales * n_facets);
  
  // For scale 0, gains is equal to gamma
  auto gains = std::vector<float>(n_facets, gamma);
  all_gains.insert(all_gains.end(), gains.begin(), gains.end());

  for (int i = 1; i < n_scales; i++) {
    auto current_psf = core::slice_leading(psfs, i);
    auto gains = compute_gain_batched(resources, stream_res, current_psf, weights_freq, gamma);
    all_gains.insert(all_gains.end(), gains.begin(), gains.end());
  }
  return all_gains;
}

}  // namespace fast_deconv::common