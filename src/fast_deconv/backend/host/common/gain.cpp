#include <algorithm>
#include <emu/submdspan.hpp>
#include <fast_deconv/common/gain.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <limits>

namespace fast_deconv::common {

std::vector<float> compute_gain_batched(const core::exec_ctx& ctx, const core::span4d<float>& psfs,
                                        const core::span1d<const float>& weights_freq, float gamma)
{
  const int n_batch = psfs.extent(0);
  const int psf_npix = psfs.extent(2) * psfs.extent(3);
  auto pmean = ctx.alloc_mdcontainer_async<float>(psfs.extent(2), psfs.extent(3));
  std::vector<float> gains(n_batch);

  for (int b = 0; b < n_batch; b++) {
    linalg::weighted_sum_async(ctx, emu::submdspan(psfs, b), weights_freq, pmean);
    const float* pmean_ptr = pmean.data_handle();
    float pmean_max = -std::numeric_limits<float>::infinity();
#pragma omp parallel for reduction(max : pmean_max)
    for (int i = 0; i < psf_npix; i++) pmean_max = std::max(pmean_max, pmean_ptr[i]);

    gains.at(b) = gamma / pmean_max;
  }

  return gains;
}

std::vector<float> compute_all_gains_batched(const core::exec_ctx& ctx, const core::span5d<float>& psfs,
                                             const core::span1d<const float>& weights_freq, float gamma)
{
  const int n_scales = psfs.extent(0);
  const int n_facets = psfs.extent(1);
  std::vector<float> all_gains;
  all_gains.reserve(n_scales * n_facets);

  // For scale 0, gains is equal to gamma
  all_gains.insert(all_gains.end(), n_facets, gamma);

  for (int i = 1; i < n_scales; i++) {
    core::span4d<float> current_psf = emu::submdspan(psfs, i);
    auto gains = compute_gain_batched(ctx, current_psf, weights_freq, gamma);
    all_gains.insert(all_gains.end(), gains.begin(), gains.end());
  }
  return all_gains;
}

}  // namespace fast_deconv::common
