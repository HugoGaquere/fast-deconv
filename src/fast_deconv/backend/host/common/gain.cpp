#include <algorithm>
#include <emu/submdspan.hpp>
#include <fast_deconv/common/gain.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <limits>

namespace fast_deconv::common {

float compute_gain(const core::exec_ctx& ctx, const float* __restrict psf, const float* __restrict weights,
                   float* __restrict scratch_buffer, int n_freq, int n, float gamma)
{
  linalg::weighted_sum_async(ctx, psf, weights, scratch_buffer, n_freq, n);
  float pmean_max = -std::numeric_limits<float>::infinity();
#pragma omp parallel for reduction(max : pmean_max)
  for (int i = 0; i < n; i++) pmean_max = std::max(pmean_max, scratch_buffer[i]);

  return gamma / pmean_max;
}

float compute_gain(const core::exec_ctx& ctx, const core::span3d<const float>& psf,
                   const core::span1d<const float>& weights_freq, float gamma)
{
  FD_PROFILE_FN();
  const int psf_npix = psf.extent(1) * psf.extent(2);
  const int n_freq = weights_freq.size();
  auto pmean = ctx.alloc_mdcontainer_async<float>(psf.extent(1), psf.extent(2));

  float gain =
      compute_gain(ctx, psf.data_handle(), weights_freq.data_handle(), pmean.data_handle(), n_freq, psf_npix, gamma);
  return gain;
}

std::vector<float> compute_gain_batched(const core::exec_ctx& ctx, const core::span4d<const float>& psfs,
                                        const core::span1d<const float>& weights_freq, float gamma)
{
  FD_PROFILE_FN();
  const int n_batch = psfs.extent(0);
  const int n_freq = psfs.extent(1);
  const int psf_npix = psfs.extent(2) * psfs.extent(3);

  auto pmean = ctx.alloc_mdcontainer_async<float>(psfs.extent(2), psfs.extent(3));
  std::vector<float> gains(n_batch);

  for (int b = 0; b < n_batch; b++) {
    auto psf = emu::submdspan(psfs, b);
    float gain =
        compute_gain(ctx, psf.data_handle(), weights_freq.data_handle(), pmean.data_handle(), n_freq, psf_npix, gamma);
    gains.at(b) = gain;
  }

  return gains;
}

}  // namespace fast_deconv::common
