#pragma once
#include <string_view>
#include <utility>
#include <vector>

namespace fast_deconv::common {

enum class convergence_status {
  running,  // in flight; never a final value
  converged,
  diverged,
  max_iterations,
  all_scales_stalled,
  no_components,
};

[[nodiscard]] std::string_view to_string(convergence_status status);

class scale_stall_tracker {
 public:
  scale_stall_tracker(int n_scales, int max_stall_count, float stall_threshold)
      : scales_stall_count_(n_scales, 0), max_stall_count_(max_stall_count), stall_threshold_(stall_threshold) {};

  void update(int scale, float rms);
  bool all_stalled() const;
  bool is_stall(int scale) const;
  void init_rms(float rms);
  std::vector<int> get_all_stalled() const;

 private:
  std::vector<int> scales_stall_count_;
  int max_stall_count_ = 0;
  float stall_threshold_ = 0.0f;
  float last_rms_ = 0.0f;
};

class convergence {
 public:
  convergence(int max_iter, float min_flux, int max_divergent_iter, float divergence_factor, scale_stall_tracker stalls)
      : stalls_(std::move(stalls)),
        max_iterations_(max_iter),
        min_flux_threshold_(min_flux),
        max_divergent_iter_(max_divergent_iter),
        divergence_factor_(divergence_factor)
  {
    flux_history_.reserve(max_iter);
    rms_history_.reserve(max_iter);
  };

  [[nodiscard]] bool should_stop() const;
  [[nodiscard]] int iteration() const;
  [[nodiscard]] convergence_status status() const;
  [[nodiscard]] bool is_stall(int scale) const;
  [[nodiscard]] std::vector<int> get_all_stalled() const;

  // Seeds the initial flux/rms; must be called once before any track().
  void init(float flux, float rms);
  void track(float flux, float rms, int subminor_count, int selected_scale);

 private:
  convergence_status status_{convergence_status::running};
  scale_stall_tracker stalls_;
  std::vector<float> flux_history_;
  std::vector<float> rms_history_;
  float divergence_factor_ = 0.0f;
  int iteration_ = 0;
  int count_divergent_iter_ = 0;

  int max_iterations_ = 0;
  int max_divergent_iter_ = 0;
  float min_flux_threshold_ = 0.0f;

  void update_status_(bool no_components);
};

}  // namespace fast_deconv::common
