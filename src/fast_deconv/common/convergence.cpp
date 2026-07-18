#include <algorithm>
#include <fast_deconv/common/convergence.hpp>
#include <ranges>

namespace fast_deconv::common {

bool convergence::should_stop() const { return status_ != convergence_status::not_converged; }

convergence_status convergence::status() const { return status_; }

void convergence::track_flux(float flux, int subminor_count)
{
  // If the flux history is empty then we are dealing with the initial flux
  if (flux_history_.empty()) {
    flux_history_.push_back(flux);
    update_status_();
    return;
  }

  // Match DDFacet: divergence iff current flux exceeds factor * previous flux,
  // and the count is cumulative across the major cycle (not reset on non-trigger).
  // TODO(guards): this only catches fast blowup. Slow exponential divergence (~1% growth per outer
  // iteration) compounds to float overflow without ever exceeding factor * previous flux. Also
  // compare against the initial flux (e.g. |flux| > divergence_factor * |flux_history_.front()|
  // => flux_diverged) to catch the slow case.
  const bool diverging_iter = std::abs(flux) > divergence_factor_ * std::abs(flux_history_.back());
  if (diverging_iter) count_divergent_iter_++;

  flux_history_.push_back(flux);
  iteration_ += subminor_count;
  update_status_();
}

int convergence::iteration() const { return iteration_; }

void convergence::update_status_()
{
  status_ = convergence_status::not_converged;
  if (iteration_ >= max_iterations_) status_ = convergence_status::max_iterations;
  if (flux_history_.back() <= min_flux_threshold_) status_ = convergence_status::flux_converged;
  if (count_divergent_iter_ > max_divergent_iter_) status_ = convergence_status::flux_diverged;
}

// ===== Scale stall tracker =====

void scale_stall_tracker::update(int scale, float rms)
{
  // Match DDFacet: cumulative stall count per scale, never reset on non-trigger.
  // TODO(guards): last_rms_ is shared across scales, so a scale gets a stall strike based on the
  // rms left by whichever scale ran before it — a plateau on one scale can retire others (observed:
  // scales 4-9 all retired within seconds). Consider tracking last_rms_ per scale so a strike only
  // reflects that scale's own progress.
  if (std::abs(last_rms_ - rms) < stall_threshold_) scales_stall_count_.at(scale)++;
  last_rms_ = rms;
}

bool scale_stall_tracker::all_stalled() const
{
  return std::ranges::all_of(scales_stall_count_, [this](int count) { return count > max_stall_count_; });
}

bool scale_stall_tracker::is_stall(int scale) const { return scales_stall_count_.at(scale) > max_stall_count_; }

void scale_stall_tracker::init_rms(float rms) { last_rms_ = rms; }

std::vector<int> scale_stall_tracker::get_all_stalled() const
{
  const int n = static_cast<int>(scales_stall_count_.size());
  auto stalled = std::ranges::views::iota(0, n) | std::ranges::views::filter([this](int s) { return is_stall(s); });
  return {stalled.begin(), stalled.end()};
}

}  // namespace fast_deconv::common