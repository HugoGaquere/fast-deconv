#include <algorithm>
#include <emu/submdspan.hpp>
#include <fast_deconv/algorithm/conv_psf_cache.hpp>
#include <fast_deconv/common/gain.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <stdexcept>
#include <utility>

namespace fast_deconv::algorithm {

conv_psf_cache::conv_psf_cache(const linalg::convolve_ctx& conv, const core::exec_ctx& reader_lane,
                               core::span4d<float> raw_psfs, core::span1d<float> sigmas, std::size_t budget_bytes)
    : conv_(conv),
      reader_lane_(reader_lane),
      raw_psfs_(raw_psfs),
      sigmas_(sigmas),
      budget_bytes_(budget_bytes),
      n_scales_(static_cast<int>(sigmas.size())),
      n_facets_(raw_psfs.extent(0)),
      n_freq_(raw_psfs.extent(1)),
      psf_nrow_(raw_psfs.extent(2)),
      psf_ncol_(raw_psfs.extent(3)),
      scratch_(conv.ctx(), conv.dims(), raw_psfs.extent(1))
{
}

std::size_t conv_psf_cache::entry_bytes(int scale) const
{
  const std::size_t plane = static_cast<std::size_t>(psf_nrow_) * psf_ncol_ * sizeof(float);
  return (scale == 0 ? 0 : plane * n_freq_) + plane;
}

void conv_psf_cache::configure(core::span1d<const float> weights, std::vector<float> weights_host, float gamma)
{
  if (weights_host_ != weights_host || gamma_ != gamma) {
    clear();
    weights_host_ = std::move(weights_host);
    gamma_ = gamma;
  }
  // The device weights are a per-call input, so they re-bind on every run.
  weights_ = weights;
}

conv_psf_cache::entry conv_psf_cache::build(int scale, int facet)
{
  FD_PROFILE_SCOPE("conv_psf_cache/build");
  const core::exec_ctx& lane = conv_.ctx();
  core::span3d<float> raw = emu::submdspan(raw_psfs_, facet);

  entry e;
  e.conv2 = lane.alloc_mdcontainer_async<float>(psf_nrow_, psf_ncol_);

  if (scale == 0) {
    // Delta kernel: conv_psf is the raw PSF itself, so alias it instead of copying.
    e.conv = core::cont3d<float>(raw.data_handle(), emu::capsule{}, raw.extents());
    linalg::weighted_sum_async(lane, core::span3d<const float>(raw), weights_, e.conv2);
    e.gain = gamma_;
  } else {
    e.conv = lane.alloc_mdcontainer_async<float>(n_freq_, psf_nrow_, psf_ncol_);
    core::span1d<float> sigma_view(sigmas_.data_handle() + scale, 1);
    scale::convolve_psf_with_scale_async(conv_, raw, sigma_view, scale, weights_, scratch_, e.conv, e.conv2);
    // compute_gain_batched wants a facet axis; this entry is that one facet.
    core::span4d<float> one_facet(e.conv.data_handle(), 1, n_freq_, psf_nrow_, psf_ncol_);
    e.gain = common::compute_gain_batched(lane, one_facet, weights_, gamma_).at(0);
  }

  // Both lanes read the entry; the build only ran on this one.
  lane.wait();
  return e;
}

void conv_psf_cache::evict_until_fits(std::size_t incoming)
{
  if (budget_bytes_ == 0 || bytes_resident_ + incoming <= budget_bytes_) return;

  // Freeing is stream-ordered on the build lane only, so drain the reader first.
  reader_lane_.wait();

  while (!entries_.empty() && bytes_resident_ + incoming > budget_bytes_) {
    auto victim = entries_.begin();
    for (auto it = entries_.begin(); it != entries_.end(); ++it)
      if (it->second.tick < victim->second.tick) victim = it;

    FD_LOG_DEBUG("conv_psf_cache: evicting scale={} facet={} ({} bytes resident)", victim->first / n_facets_,
                 victim->first % n_facets_, bytes_resident_);
    bytes_resident_ -= entry_bytes(victim->first / n_facets_);
    entries_.erase(victim);
    stats_.evictions++;
  }
}

conv_psf_cache::entry conv_psf_cache::get(int scale, int facet)
{
  if (weights_host_.empty()) throw std::logic_error("conv_psf_cache::get called before configure()");
  if (scale < 0 || scale >= n_scales_ || facet < 0 || facet >= n_facets_)
    throw std::out_of_range("conv_psf_cache::get: (scale, facet) out of range");

  const int key = scale * n_facets_ + facet;
  if (auto it = entries_.find(key); it != entries_.end()) {
    it->second.tick = ++tick_;
    stats_.hits++;
    return it->second.value;
  }

  stats_.misses++;
  entry e = build(scale, facet);

  const std::size_t bytes = entry_bytes(scale);
  // A budget smaller than one entry would evict everything and still not fit;
  // hand the entry over uncached rather than thrash.
  if (budget_bytes_ != 0 && bytes > budget_bytes_) return e;

  evict_until_fits(bytes);
  entries_.emplace(key, slot{e, ++tick_});
  bytes_resident_ += bytes;
  stats_.bytes_peak = std::max(stats_.bytes_peak, bytes_resident_);
  return e;
}

void conv_psf_cache::prefetch_scale(int scale)
{
  FD_PROFILE_SCOPE_FMT("conv_psf_cache/prefetch_scale[{}]", scale);
  for (int f = 0; f < n_facets_; f++) get(scale, f);
}

void conv_psf_cache::prefetch_all()
{
  FD_PROFILE_SCOPE("conv_psf_cache/prefetch_all");
  for (int s = 0; s < n_scales_; s++) prefetch_scale(s);
}

void conv_psf_cache::set_budget_bytes(std::size_t bytes)
{
  budget_bytes_ = bytes;
  evict_until_fits(0);
}

void conv_psf_cache::clear()
{
  if (entries_.empty()) return;
  reader_lane_.wait();
  conv_.ctx().wait();
  entries_.clear();
  bytes_resident_ = 0;
}

}  // namespace fast_deconv::algorithm
