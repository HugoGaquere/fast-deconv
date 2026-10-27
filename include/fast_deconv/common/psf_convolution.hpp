#pragma once

#include <cstdint>
#include <fast_deconv/common/scales.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/gaussian_convolution.hpp>
#include <unordered_map>
#include <vector>

namespace fast_deconv::common {

class psf_convolution {
 public:
  struct entry {
    core::cont3d<const float> conv;   // (n_freq, psf_h, psf_w); aliases the raw PSF at scale 0
    core::cont2d<const float> conv2;  // (psf_h, psf_w), weighted mean over channels
    float gain = 0.0f;
  };

  /**
   * @brief Lazy cache of the scale-convolved PSFs of one scale, built per (scale, facet) on first use.
   *
   * @param[in] exec_ctx  Context the builds run on; each build waits on it before returning. Must outlive this.
   * @param[in] raw_psfs  Raw PSFs, device, (n_facets, n_freq, psf_h, psf_w). Must outlive this.
   * @param[in] sigmas    Scale sigmas, host, (n_scales,). Copied.
   * @param[in] weights   Per-channel weights, device, (n_freq,). Must outlive this.
   * @param[in] gamma     CLEAN loop gain, folded into each entry's gain.
   */
  psf_convolution(const core::exec_ctx& exec_ctx, core::span4d<const float> raw_psfs, std::vector<float> sigmas,
                  core::span1d<const float> weights, float gamma);

  psf_convolution(const psf_convolution&) = delete;
  psf_convolution& operator=(const psf_convolution&) = delete;
  psf_convolution(psf_convolution&&) = delete;
  psf_convolution& operator=(psf_convolution&&) = delete;

  /**
   * @brief Entry for one (scale, facet), built on miss.
   *
   * A new scale frees the cached entries: sync every context still reading them first.
   */
  entry get(int scale, std::int64_t facet);

  std::size_t n_entries() const { return entries_.size(); }

 private:
  entry build(int scale, std::int64_t facet);

  const core::exec_ctx& exec_ctx_;
  const linalg::gaussian_convolution_ctx conv_ctx_;       // batch n_freq: the per-channel conv
  const linalg::gaussian_convolution_ctx mean_conv_ctx_;  // batch 1: conv2 of the channel-weighted mean PSF
  core::span4d<const float> raw_psfs_;
  core::span1d<const float> weights_;
  std::vector<float> sigmas_;

  int n_scales_;
  std::int64_t n_facets_;
  std::int64_t n_freq_;
  std::int64_t psf_nrow_;
  std::int64_t psf_ncol_;

  float gamma_ = 0.0f;
  int scale_ = -1;  // scale the entries belong to

  std::unordered_map<std::int64_t, entry> entries_;  // scale_ entries, keyed by facet
};

}  // namespace fast_deconv::common
