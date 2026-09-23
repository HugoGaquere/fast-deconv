#pragma once

#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/convolution.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <unordered_map>
#include <vector>

namespace fast_deconv::algorithm {

enum class psf_cache_mode {
  lazy_pair,   // build one (scale, facet) on first use
  lazy_scale,  // build every facet of a scale on that scale's first use
  eager_all,   // build every (scale, facet) up front
};

class psf_convolution {
 public:
  struct entry {
    core::cont3d<const float> conv;   // (n_freq, psf_h, psf_w); aliases the raw PSF at scale 0
    core::cont2d<const float> conv2;  // (psf_h, psf_w), weighted mean over channels
    float gain = 0.0f;
  };

  /**
   * @param exec_ctx  Lane the builds run on; each build waits on it before returning. Must outlive this.
   * @param raw_psfs  Raw PSFs, device, (n_facets, n_freq, psf_h, psf_w). Must outlive this.
   * @param sigmas    Scale sigmas, host, (n_scales,). Copied.
   * @param weights   Per-channel weights, device, (n_freq,). Must outlive this.
   * @param gamma     CLEAN loop gain, folded into each entry's gain.
   * @param padding   FFT padding factor for the PSF-grid convolutions.
   */
  psf_convolution(const core::exec_ctx& exec_ctx, core::span4d<const float> raw_psfs, std::vector<float> sigmas,
                  core::span1d<const float> weights, float gamma, float padding);

  psf_convolution(const psf_convolution&) = delete;
  psf_convolution& operator=(const psf_convolution&) = delete;
  psf_convolution(psf_convolution&&) = delete;
  psf_convolution& operator=(psf_convolution&&) = delete;

  /// Entry for one (scale, facet), built on miss.
  entry get(int scale, int facet);

  /// Build every facet of @p scale.
  void prefetch_scale(int scale);

  /// Build every (scale, facet).
  void prefetch_all();

  /// Drops every entry, once the build lane has drained. The caller must first drain
  /// any other lane still reading handed-out entries: the frees are ordered on the build lane only.
  void clear();

  int n_scales() const { return n_scales_; }
  int n_facets() const { return n_facets_; }
  std::size_t n_entries() const { return entries_.size(); }

 private:
  entry build(int scale, int facet);

  const core::exec_ctx& exec_ctx_;
  const linalg::convolution_ctx conv_ctx_;
  core::span4d<const float> raw_psfs_;
  core::span1d<const float> weights_;
  std::vector<float> sigmas_;

  int n_scales_;
  int n_facets_;
  int n_freq_;
  int psf_nrow_;
  int psf_ncol_;

  float gamma_ = 0.0f;

  std::unordered_map<int, entry> entries_;  // key = scale * n_facets + facet
};

}  // namespace fast_deconv::algorithm
