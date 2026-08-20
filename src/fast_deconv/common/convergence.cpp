#include <algorithm>
#include <cmath>
#include <fast_deconv/common/convergence.hpp>
#include <ranges>

namespace fast_deconv::common {

std::string_view to_string(convergence_status status)
{
  switch (status) {
    case convergence_status::running:
      return "running";
    case convergence_status::converged:
      return "converged";
    case convergence_status::diverged:
      return "diverged";
    case convergence_status::max_iterations:
      return "max_iterations";
    case convergence_status::all_scales_stalled:
      return "all_scales_stalled";
    case convergence_status::no_components:
      return "no_components";
  }
  return "unknown";
}

bool convergence::should_stop() const { return status_ != convergence_status::running; }

convergence_status convergence::status() const { return status_; }

bool convergence::is_stall(int scale) const { return stalls_.is_stall(scale); }

std::vector<int> convergence::get_all_stalled() const { return stalls_.get_all_stalled(); }

void convergence::init(float flux, float rms)
{
  flux_history_.push_back(flux);
  rms_history_.push_back(rms);
  stalls_.init_rms(rms);
  update_status_(false);
}

void convergence::track(float flux, float rms, int subminor_count, int selected_scale)
{
  // Match DDFacet: divergence iff current flux exceeds factor * previous flux,
  // and the count is cumulative across the major cycle (not reset on non-trigger).
  // The initial-flux test catches slow growth that never exceeds factor * previous flux.
  const bool diverging_iter = std::abs(flux) > divergence_factor_ * std::abs(flux_history_.back()) ||
                              std::abs(flux) > divergence_factor_ * std::abs(flux_history_.front());
  if (diverging_iter) count_divergent_iter_++;

  flux_history_.push_back(flux);
  rms_history_.push_back(rms);
  iteration_ += subminor_count;
  stalls_.update(selected_scale, rms);
  update_status_(subminor_count == 0);
}

int convergence::iteration() const { return iteration_; }

void convergence::update_status_(bool no_components)
{
  // Written weakest-reason first: the checks overwrite, so the last one that holds is reported.
  status_ = convergence_status::running;
  if (iteration_ >= max_iterations_) status_ = convergence_status::max_iterations;
  if (stalls_.all_stalled()) status_ = convergence_status::all_scales_stalled;
  if (no_components) status_ = convergence_status::no_components;
  if (flux_history_.back() <= min_flux_threshold_) status_ = convergence_status::converged;
  if (count_divergent_iter_ > max_divergent_iter_) status_ = convergence_status::diverged;
  // Last: every ordering test above is false on NaN, so an overflowed residual reads as running.
  if (!std::isfinite(flux_history_.back()) || !std::isfinite(rms_history_.back()))
    status_ = convergence_status::diverged;
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