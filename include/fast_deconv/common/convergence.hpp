#pragma once
#include <string_view>
#include <utility>
#include <vector>

namespace fast_deconv::common {

enum class convergence_status {
  running,             // in flight; never a final value
  converged,           // peak flux at or below the stop flux
  diverged,            // too many divergent iterations, or a non-finite flux or standard deviation
  max_iterations,      // minor-iteration budget spent
  all_scales_stalled,  // every scale retired by the stall tracker
  no_components,       // the last scale selection found nothing to subtract
};

/// Lower-case name of @p status, as logged.
[[nodiscard]] std::string_view to_string(convergence_status status);

/**
 * @brief Per-scale stall strikes, from the change in standard deviation between outer iterations.
 *
 * Each update() compares against the previous update(), whichever scale it was for. A scale is stalled once its
 * consecutive strikes exceed max_stall_count.
 */
class scale_stall_tracker {
 public:
  /**
   * @param[in] n_scales         Number of scales tracked.
   * @param[in] max_stall_count  Strikes a scale may take before it counts as stalled.
   * @param[in] stall_threshold  Relative standard deviation change below which an update is a strike.
   */
  scale_stall_tracker(int n_scales, int max_stall_count, float stall_threshold)
      : scales_stall_count_(n_scales, 0), max_stall_count_(max_stall_count), stall_threshold_(stall_threshold) {};

  /// Strike for @p scale if @p rms (standard deviation) moved by less than the threshold times the previous one,
  /// else reset its count.
  void update(int scale, float rms);
  /// True once every scale is stalled.
  bool all_stalled() const;
  /// True once @p scale has more than max_stall_count consecutive strikes.
  bool is_stall(int scale) const;
  /// Sets the standard deviation the first update() compares against.
  void init_rms(float rms);
  /// Indices of the stalled scales, ascending.
  std::vector<int> get_all_stalled() const;

 private:
  std::vector<int> scales_stall_count_;
  int max_stall_count_ = 0;
  float stall_threshold_ = 0.0f;
  float last_rms_ = 0.0f;
};

/**
 * @brief Stop decision for the outer loop: convergence, divergence, iteration budget and scale stalls.
 *
 * When several reasons hold at once, status() reports the strongest: a non-finite value, then diverged,
 * converged, no_components, all_scales_stalled and max_iterations.
 */
class convergence {
 public:
  /**
   * @param[in] max_iter            Minor-iteration budget across all outer iterations.
   * @param[in] min_flux            Stop flux: converged once the peak flux is at or below it.
   * @param[in] max_divergent_iter  Divergent iterations tolerated, counted cumulatively; one more is diverged.
   * @param[in] divergence_factor   Divergent when |flux| exceeds this times the previous or the initial |flux|.
   * @param[in] stalls              Stall tracker, updated by every track().
   */
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

  /// True once status() is anything but running.
  [[nodiscard]] bool should_stop() const;
  /// Minor iterations consumed so far.
  [[nodiscard]] int iteration() const;
  [[nodiscard]] convergence_status status() const;
  /// See scale_stall_tracker::is_stall().
  [[nodiscard]] bool is_stall(int scale) const;
  /// See scale_stall_tracker::get_all_stalled().
  [[nodiscard]] std::vector<int> get_all_stalled() const;

  /// Seeds the initial flux and standard deviation; must be called once before any track().
  void init(float flux, float rms);

  /**
   * @brief Record one outer iteration and update status().
   *
   * @param[in] flux            Peak flux of the mean residual after the iteration.
   * @param[in] rms             Standard deviation of the mean residual after the iteration.
   * @param[in] subminor_count  Minor iterations the iteration ran; 0 means no component was found.
   * @param[in] selected_scale  Scale the iteration cleaned with; takes the stall update.
   */
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
