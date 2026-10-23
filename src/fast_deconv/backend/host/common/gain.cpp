#include <algorithm>
#include <cstdint>
#include <emu/submdspan.hpp>
#include <fast_deconv/common/gain.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <limits>

namespace fast_deconv::common {

namespace {

// gamma / max of the weighted channel mean of @p psf, built in @p pmean.
float gain_of(const core::exec_ctx& ctx, core::span3d<const float> psf, core::span1d<const float> weights,
              core::span2d<float> pmean, float gamma)
{
  linalg::weighted_sum_async(ctx, psf, weights, pmean);
  const float* p = pmean.data_handle();
  const std::int64_t n = pmean.mapping().required_span_size();
  float pmean_max = -std::numeric_limits<float>::infinity();
#pragma omp parallel for reduction(max : pmean_max)
  for (std::int64_t i = 0; i < n; i++) pmean_max = std::max(pmean_max, p[i]);

  return gamma / pmean_max;
}

}  // namespace

float compute_gain(const core::exec_ctx& ctx, core::span3d<const float> psf, core::span1d<const float> weights_freq,
                   float gamma)
{
  FD_PROFILE_FN();
  auto pmean = ctx.alloc_mdcontainer_async<float>(psf.extent(1), psf.extent(2));
  return gain_of(ctx, psf, weights_freq, pmean, gamma);
}

std::vector<float> compute_gain_batched(const core::exec_ctx& ctx, core::span4d<const float> psfs,
                                        core::span1d<const float> weights_freq, float gamma)
{
  FD_PROFILE_FN();
  const std::int64_t n_batch = psfs.extent(0);

  auto pmean = ctx.alloc_mdcontainer_async<float>(psfs.extent(2), psfs.extent(3));
  std::vector<float> gains(core::narrow<std::size_t>(n_batch));

  for (std::int64_t b = 0; b < n_batch; b++) {
    const core::span3d<const float> psf = emu::submdspan(psfs, b);
    gains.at(core::narrow<std::size_t>(b)) = gain_of(ctx, psf, weights_freq, pmean, gamma);
  }

  return gains;
}

}  // namespace fast_deconv::common
