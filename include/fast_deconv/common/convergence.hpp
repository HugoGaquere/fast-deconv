#pragma once
#include <vector>

namespace fast_deconv::common {

enum class convergence_status {
  not_converged,
  flux_converged,
  flux_diverged,
  // stalled,         // all scales stalled (RMS change below threshold)
  max_iterations,  // reached max_iteration count
};

class convergence {
 public:
  convergence(int max_iter, float min_flux, int max_divergent_iter, float divergence_factor)
      : max_iterations_(max_iter),
        min_flux_threshold_(min_flux),
        max_divergent_iter_(max_divergent_iter),
        divergence_factor_(divergence_factor)
  {
    flux_history_.reserve(max_iter);
  };

  [[nodiscard]] bool should_stop() const;
  [[nodiscard]] int iteration() const;
  [[nodiscard]] convergence_status status() const;
  void track_flux(float flux, int subminor_count = 1);

 private:
  convergence_status status_{convergence_status::not_converged};
  std::vector<float> flux_history_;
  float divergence_factor_ = 0.0f;
  int iteration_ = 0;
  int count_divergent_iter_ = 0;

  int max_iterations_ = 0;
  int max_divergent_iter_ = 0;
  float min_flux_threshold_ = 0.0f;

  void update_status_();
};

class scale_stall_tracker {
 public:
  scale_stall_tracker(int n_scales, int max_stall_count, float stall_threshold)
      : scales_stall_count_(n_scales, 0),
        max_stall_count_(max_stall_count),
        stall_threshold_(stall_threshold) {};

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

}  // namespace fast_deconv::common
