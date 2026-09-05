#pragma once

#include <cstddef>
#include <cstdint>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <unordered_map>
#include <vector>

namespace fast_deconv::algorithm {

/// How eagerly the cache fills itself. Every mode goes through the same build
/// path, so they differ only in when the misses happen.
enum class psf_cache_mode {
  lazy_pair,   // build one (scale, facet) on first use
  lazy_scale,  // build every facet of a scale on that scale's first use
  eager_all,   // build every (scale, facet) up front
};

/**
 * @brief   Owns the convolved PSFs for the whole deconvolution session.
 * @details The clean loop needs conv_psf/conv2_psf/gain for one (scale, facet)
 *          at a time, but only a fraction of the n_scales * n_facets pairs is
 *          ever touched. Entries are therefore built on demand and held under a
 *          byte budget, with the least recently used one evicted first.
 *
 *          Scale 0 is the delta kernel: its conv_psf *is* the raw PSF, so the
 *          entry aliases it instead of copying, and its gain is gamma.
 *
 *          The entries are a pure function of the raw PSFs and sigmas (session
 *          constants) plus the frequency weights and gamma, so configure()
 *          drops everything when either of the latter two moves.
 */
class conv_psf_cache {
 public:
  /// Convolved PSFs for one (scale, facet). Copies share the buffers through
  /// emu::capsule, so a copy handed out by get() outlives an eviction.
  struct entry {
    core::cont3d<float> conv;   // (n_freq, psf_h, psf_w); aliases the raw PSF at scale 0
    core::cont2d<float> conv2;  // (psf_h, psf_w), weighted mean over channels
    float gain = 0.0f;
  };

  struct stats {
    int hits = 0;
    int misses = 0;
    int evictions = 0;
    std::size_t bytes_peak = 0;
  };

  /**
   * @param conv         PSF-grid convolution plans; its lane is where builds run.
   * @param reader_lane  The other lane that reads handed-out entries. Drained before
   *                     an eviction frees memory, since the free is only stream-ordered
   *                     against the build lane.
   * @param raw_psfs     Raw PSFs, device, (n_facets, n_freq, psf_h, psf_w). Must outlive this.
   * @param sigmas       Scale sigmas, device, (n_scales,). Must outlive this.
   * @param budget_bytes Cap on resident entry bytes; 0 means unbounded.
   */
  conv_psf_cache(const linalg::convolve_ctx& conv, const core::exec_ctx& reader_lane, core::span4d<float> raw_psfs,
                 core::span1d<float> sigmas, std::size_t budget_bytes);

  conv_psf_cache(const conv_psf_cache&) = delete;
  conv_psf_cache& operator=(const conv_psf_cache&) = delete;
  conv_psf_cache(conv_psf_cache&&) = delete;
  conv_psf_cache& operator=(conv_psf_cache&&) = delete;

  /// Rebinds the per-call inputs the entries depend on, clearing the cache if
  /// either changed. Must be called before the first get() of a run.
  void configure(core::span1d<const float> weights, std::vector<float> weights_host, float gamma);

  /// Entry for one (scale, facet), built on miss. The returned copy stays valid
  /// even if a later call evicts it from the cache.
  entry get(int scale, int facet);

  /// Build every facet of @p scale.
  void prefetch_scale(int scale);

  /// Build every (scale, facet). Only meaningful with an unbounded budget.
  void prefetch_all();

  /// Drops every entry. Frees on the build lane, after draining the reader lane.
  void clear();

  /// Changes the resident cap, evicting down to it immediately. 0 = unbounded.
  void set_budget_bytes(std::size_t bytes);

  std::size_t bytes_resident() const { return bytes_resident_; }
  std::size_t budget_bytes() const { return budget_bytes_; }
  /// Bytes one entry of @p scale occupies; scale 0 only owns its conv2 plane.
  std::size_t entry_bytes(int scale) const;
  int n_scales() const { return n_scales_; }
  int n_facets() const { return n_facets_; }
  const stats& get_stats() const { return stats_; }

 private:
  entry build(int scale, int facet);
  void evict_until_fits(std::size_t incoming);

  const linalg::convolve_ctx& conv_;
  const core::exec_ctx& reader_lane_;
  core::span4d<float> raw_psfs_;
  core::span1d<float> sigmas_;
  std::size_t budget_bytes_;

  int n_scales_;
  int n_facets_;
  int n_freq_;
  int psf_nrow_;
  int psf_ncol_;

  core::span1d<const float> weights_;
  std::vector<float> weights_host_;
  float gamma_ = 0.0f;

  struct slot {
    entry value;
    std::uint64_t tick;  // LRU stamp, bumped on every hit
  };

  std::unordered_map<int, slot> entries_;  // key = scale * n_facets + facet
  std::uint64_t tick_ = 0;
  std::size_t bytes_resident_ = 0;
  scale::psf_convolve_scratch scratch_;
  stats stats_;
};

}  // namespace fast_deconv::algorithm
