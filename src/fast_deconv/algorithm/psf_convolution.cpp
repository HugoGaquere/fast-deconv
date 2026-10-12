#include <emu/submdspan.hpp>
#include <fast_deconv/algorithm/psf_convolution.hpp>
#include <fast_deconv/common/gain.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <numbers>
#include <stdexcept>

namespace fast_deconv::algorithm {

psf_convolution::psf_convolution(const core::exec_ctx& exec_ctx, core::span4d<const float> raw_psfs,
                                 std::vector<float> sigmas, core::span1d<const float> weights, float gamma)
    : exec_ctx_(exec_ctx),
      // Wrap-free for every scale: conv uses sigma, conv2 sigma * sqrt(2).
      conv_ctx_(exec_ctx, raw_psfs.extent(1), raw_psfs.extent(2), raw_psfs.extent(3),
                linalg::max_gaussian_reach(sigmas)),
      mean_conv_ctx_(exec_ctx, 1, raw_psfs.extent(2), raw_psfs.extent(3),
                     linalg::max_gaussian_reach(sigmas, std::numbers::sqrt2_v<float>)),
      raw_psfs_(raw_psfs),
      weights_(weights),
      sigmas_(std::move(sigmas)),
      n_scales_(static_cast<int>(sigmas_.size())),
      n_facets_(raw_psfs.extent(0)),
      n_freq_(raw_psfs.extent(1)),
      psf_nrow_(raw_psfs.extent(2)),
      psf_ncol_(raw_psfs.extent(3)),
      gamma_(gamma)
{
}

psf_convolution::entry psf_convolution::build(int scale, int facet)
{
  FD_PROFILE_SCOPE("psf_convolution/build");
  core::span3d<const float> raw = emu::submdspan(raw_psfs_, facet);

  entry e;
  // core::cont3d<float> conv;
  auto conv2_mean = exec_ctx_.alloc_mdcontainer_async<float>(psf_nrow_, psf_ncol_);

  if (scale == 0) {
    // Delta kernel: conv_psf is the raw PSF itself, so alias it instead of copying.
    e.conv = core::cont3d<const float>(raw.data_handle(), emu::capsule{}, raw.extents());
    linalg::weighted_sum_async(exec_ctx_, raw, weights_, conv2_mean);
    e.gain = gamma_;
  } else {
    // Previous version: every channel convolved once and twice, then the weighted mean of the twice-convolved.
    // auto conv = exec_ctx_.alloc_mdcontainer_async<float>(n_freq_, psf_nrow_, psf_ncol_);
    // auto conv2 = exec_ctx_.alloc_mdcontainer_async<float>(n_freq_, psf_nrow_, psf_ncol_);
    // conv_ctx_.convolve_with_gaussian_once_and_twice(raw, sigmas_.at(scale), conv, conv2);
    // linalg::weighted_sum_async(exec_ctx_, conv2, weights_, conv2_mean);
    const float sigma = sigmas_.at(scale);
    auto conv = exec_ctx_.alloc_mdcontainer_async<float>(n_freq_, psf_nrow_, psf_ncol_);
    {
      FD_PROFILE_SCOPE("psf_convolution/conv");
      conv_ctx_.convolve(raw, sigma, conv);
    }
    // Only the channel-weighted mean of conv2 is kept. G(sigma) twice is G(sigma * sqrt(2)) (the spectra
    // multiply: H(sigma)^2 = H(sigma * sqrt(2))), and convolution is linear, so
    // sum_f w_f (G*G * psf_f) = G(sigma * sqrt(2)) * (sum_f w_f psf_f): one plane, one convolution.
    {
      FD_PROFILE_SCOPE("psf_convolution/conv2");
      auto raw_mean = exec_ctx_.alloc_mdcontainer_async<float>(psf_nrow_, psf_ncol_);
      linalg::weighted_sum_async(exec_ctx_, raw, weights_, raw_mean);
      mean_conv_ctx_.convolve(core::span3d<const float>(raw_mean.data_handle(), 1, psf_nrow_, psf_ncol_),
                              sigma * std::numbers::sqrt2_v<float>,
                              core::span3d<float>(conv2_mean.data_handle(), 1, psf_nrow_, psf_ncol_));
    }
    e.gain = common::compute_gain(exec_ctx_, conv, weights_, gamma_);
    e.conv = conv;
  }
  e.conv2 = conv2_mean;

  exec_ctx_.wait();
  return e;
}

psf_convolution::entry psf_convolution::get(int scale, int facet)
{
  FD_PROFILE_SCOPE("psf_convolution/get");
  // if (weights_host_.empty()) throw std::logic_error("psf_convolution::get called before configure()");
  if (scale < 0 || scale >= n_scales_ || facet < 0 || facet >= n_facets_)
    throw std::out_of_range("psf_convolution::get: (scale, facet) out of range");

  const int key = scale * n_facets_ + facet;
  if (auto it = entries_.find(key); it != entries_.end()) {
    return it->second;
  }

  entry e = build(scale, facet);
  entries_.emplace(key, e);
  return e;
}

void psf_convolution::prefetch_scale(int scale)
{
  FD_PROFILE_SCOPE_FMT("psf_convolution/prefetch_scale[{}]", scale);
  for (int f = 0; f < n_facets_; f++) get(scale, f);
}

void psf_convolution::prefetch_all()
{
  FD_PROFILE_SCOPE("psf_convolution/prefetch_all");
  for (int s = 0; s < n_scales_; s++) prefetch_scale(s);
}

void psf_convolution::clear()
{
  if (entries_.empty()) return;
  exec_ctx_.wait();
  entries_.clear();
}

}  // namespace fast_deconv::algorithm
