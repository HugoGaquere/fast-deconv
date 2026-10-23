#include <cstdint>
#include <emu/submdspan.hpp>
#include <fast_deconv/common/gain.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <fast_deconv/matrix/argmax.hpp>

namespace fast_deconv::common {

namespace {

// gamma / max of the weighted channel mean of @p psf, built in @p pmean.
float gain_of(const core::exec_ctx& ctx, core::span3d<const float> psf, core::span1d<const float> weights,
              core::span2d<float> pmean, float gamma)
{
  linalg::weighted_sum(ctx, psf, weights, pmean);
  return gamma / matrix::find_peak(ctx, pmean, {}).value;
}

}  // namespace

float compute_gain(const core::exec_ctx& ctx, core::span3d<const float> psf, core::span1d<const float> weights_freq,
                   float gamma)
{
  FD_PROFILE_FN();
  auto pmean = ctx.alloc_mdcontainer<float>(psf.extent(1), psf.extent(2));
  return gain_of(ctx, psf, weights_freq, pmean, gamma);
}

std::vector<float> compute_gain_batched(const core::exec_ctx& ctx, core::span4d<const float> psfs,
                                        core::span1d<const float> weights_freq, float gamma)
{
  FD_PROFILE_FN();
  const std::int64_t n_batch = psfs.extent(0);

  auto pmean = ctx.alloc_mdcontainer<float>(psfs.extent(2), psfs.extent(3));
  std::vector<float> gains(core::narrow<std::size_t>(n_batch));

  for (std::int64_t b = 0; b < n_batch; b++) {
    const core::span3d<const float> psf = emu::submdspan(psfs, b);
    gains.at(core::narrow<std::size_t>(b)) = gain_of(ctx, psf, weights_freq, pmean, gamma);
  }

  return gains;
}

}  // namespace fast_deconv::common
