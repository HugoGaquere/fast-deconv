#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <emu/submdspan.hpp>
#include <fast_deconv/algorithm/ddmsc.hpp>
#include <fast_deconv/common/convergence.hpp>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/common/multi_frequency.hpp>
#include <fast_deconv/common/psf_convolution.hpp>
#include <fast_deconv/common/scales.hpp>
#include <fast_deconv/common/subtract.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <fast_deconv/matrix/stats.hpp>
#include <fast_deconv/matrix/tiled_argmax.hpp>
#include <fast_deconv/util/utils.hpp>
#include <optional>
#include <stdexcept>
#include <vector>

namespace fast_deconv::algorithm::ddmsc {

namespace {

inline std::string fmt_optional(const std::optional<float>& v)
{
  return v.has_value() ? fmt::format("{:g}", *v) : std::string{"unset"};
}

std::string format_run_banner(const params& p, std::int64_t dirty_nrows, std::int64_t dirty_ncols, std::int64_t n_freq,
                              std::int64_t n_facets, int n_scales, std::int64_t psf_nrow, std::int64_t psf_ncol)
{
  constexpr const char* sep = "------------------------------------------------------------------------";
  return fmt::format(
      "run_ddmsc: launching deconvolution\n"
      "  {}\n"
      "  image      | dirty={}x{}  n_freq={}  n_facets={}  psf={}x{}\n"
      "  scales     | n_scales={}  stall_threshold={:g}\n"
      "  outer loop | max_iteration={}  divergence_factor={:.4f}\n"
      "  stop crit  | flux_threshold={:.6e}  rms_factor={:.4f}  peak_factor={:.4f}\n"
      "             | cycle_factor={:.4f}  sidelobe_level={:.4f}\n"
      "  clean loop | max_clean_iteration={}  peak_factor={:.4f}  gamma={:.4f}  clean_negative={}\n"
      "  auto mask  | enable={}  force={}  peak_threshold={}  rms_threshold={}\n"
      "  {}",
      sep, dirty_nrows, dirty_ncols, n_freq, n_facets, psf_nrow, psf_ncol, n_scales, p.scale_stall_threshold,
      p.max_iteration, p.divergence_factor, p.flux_threshold, p.stop_rms_factor, p.stop_peak_factor,
      p.stop_cycle_factor, p.stop_sidelobe_level, p.max_clean_iteration, p.peak_factor, p.gamma, p.clean_negative,
      p.enable_auto_mask, p.force_enable_auto_mask, fmt_optional(p.auto_mask_peak_threshold),
      fmt_optional(p.auto_mask_rms_threshold), sep);
}

struct stop_limits {
  float rms;
  float peak;
  float sidelobe;
  float stop_flux;
};

stop_limits compute_stop_flux(const params& p, float flux, float rms)
{
  const float sidelobe_coeff =
      (p.stop_cycle_factor != 0.0f)
          ? ((p.stop_cycle_factor - 1.0f) / 4.0f * (1.0f - p.stop_sidelobe_level) + p.stop_sidelobe_level)
          : 0.0f;
  stop_limits limits{
      .rms = p.stop_rms_factor * rms, .peak = p.stop_peak_factor * flux, .sidelobe = sidelobe_coeff * flux};
  limits.stop_flux = std::max({p.flux_threshold, limits.rms, limits.peak, limits.sidelobe});
  return limits;
}

std::optional<float> auto_mask_trigger(const params& p, float flux, float rms)
{
  const float threshold = p.auto_mask_peak_threshold.value_or(p.auto_mask_rms_threshold.value_or(0.f) * rms);
  if (p.enable_auto_mask && (p.force_enable_auto_mask || flux <= threshold)) return threshold;
  return std::nullopt;
}

struct run_workspace {
  core::cont2d<float> mean_residual_buf;
  matrix::tiled_argmax_ctx tiled;
  const linalg::gaussian_convolution_ctx scale_conv;
  common::psf_convolution psf_cache;
  // Coefficient buffers live on cube_exec, which writes them.
  core::cont1d<float> all_coeffs;
  core::cont1d<float> coeffs_per_chan;
  // Built once, the first time the auto-mask engages.
  std::optional<core::cont3d<bool>> mask_per_scale;
  // Pixels some active (non-retired) scale may clean; the stop and divergence stats are measured on it.
  std::optional<core::cont2d<bool>> merged_active_mask;

  run_workspace(const context& ctx, const params& p, std::int64_t nrow, std::int64_t ncol, std::int64_t n_freq,
                core::span1d<const float> weights_freq)
      : mean_residual_buf(ctx.exec.alloc_mdcontainer<float>(nrow, ncol)),
        tiled(ctx.exec, mean_residual_buf.extents(), 64),
        scale_conv(ctx.exec, 1, ctx.dirty_nrow, ctx.dirty_ncol, ctx.scale_sigmas),
        psf_cache(ctx.exec, ctx.raw_psfs, ctx.scale_sigmas, weights_freq, p.gamma),
        all_coeffs(ctx.cube_exec.alloc_mdcontainer<float>((std::int64_t{p.max_iteration} + p.max_clean_iteration) *
                                                          ctx.xdes.extent(1))),
        coeffs_per_chan(ctx.cube_exec.alloc_mdcontainer<float>(n_freq))
  {
  }
};

// Per-scale masks around every component so far, previous calls included.
core::cont3d<bool> build_scale_mask(const context& ctx, const ddmsc_result& result,
                                    core::span1d<const float> weights_freq, std::int64_t nrow, std::int64_t ncol)
{
  FD_PROFILE_FN();
  auto mask_per_scale = ctx.exec.alloc_mdcontainer<bool>(ctx.scale_sigmas.size(), nrow, ncol);

  const int central_facet_idx = ctx.map_pixel_facet(nrow / 2, ncol / 2);
  core::span3d<const float> central_facet_psfs = emu::submdspan(ctx.raw_psfs, central_facet_idx);

  std::vector<core::index2d> all_coords = ctx.historical_peak_coords;
  for (const auto& [r, c] : result.peak_coords) all_coords.push_back({r, c});
  std::vector<int> all_scales = ctx.historical_scales;
  all_scales.insert(all_scales.end(), result.scales.begin(), result.scales.end());

  common::build_auto_mask(ctx.exec, all_coords, all_scales, central_facet_psfs, weights_freq, ctx.scale_sigmas,
                          ctx.mask, mask_per_scale);
  return mask_per_scale;
}

// Returns the number of components found, with both contexts synced.
int clean_loop(run_workspace& ws, const context& ctx, const params& p, const scale::scale_result& selected,
               core::span3d<float> dirty, core::span3d<const float> jones_norm, core::span1d<const float> weights_freq,
               int first_component, ddmsc_result& result)
{
  FD_PROFILE_FN();
  const std::int64_t dirty_ncols = dirty.extent(2);
  const std::int64_t n_order = ctx.xdes.extent(1);
  const int selected_scale_idx = selected.scale;
  const core::span2d<float> scaled_residual = selected.scaled_residual;

  auto [peak_index, peak_value, peak_signed] = selected.peak;

  const float threshold = peak_value * p.peak_factor;

  FD_PROFILE_PLOT("scale", static_cast<std::int64_t>(selected_scale_idx));
  FD_PROFILE_PLOT("scale_peak", peak_value);

  // Seed only: the scale search already found the first peak.
  ws.tiled.run(scaled_residual, selected.criterion);

  FD_LOG_DEBUG("run_ddmsc: clean loop start scale={} peak={:.8f} threshold={:.8f} max_clean_iter={}",
               selected_scale_idx, peak_value, threshold, p.max_clean_iteration);

  int n_clean_iter = 0;
  while (peak_value > threshold && n_clean_iter < p.max_clean_iteration) {
    FD_PROFILE_SCOPE("minor_iter");
    const auto peak_coords = util::unravel_index_2D(peak_index, dirty_ncols);
    const int facet_idx = ctx.map_pixel_facet(peak_coords.row, peak_coords.col);
    const common::psf_convolution::entry psf = ws.psf_cache.get(selected_scale_idx, facet_idx);
    const float gain = psf.gain;

    result.add_component(peak_coords, selected_scale_idx, gain);

    FD_LOG_DEBUG("run_ddmsc:   [sub={}] peak={:.8f} at ({},{}) facet={} gain={:.6f}", n_clean_iter, peak_value,
                 peak_coords.row, peak_coords.col, facet_idx, gain);

    core::span3d<const float> conv_psf = psf.conv;
    core::span2d<const float> conv2_psf = psf.conv2;

    const std::int64_t coeffs_offset = (std::int64_t{first_component} + n_clean_iter) * n_order;
    auto spectral_coeffs = core::span1d<float>(ws.all_coeffs.data_handle() + coeffs_offset, n_order);
    multi_frequency::fit_coefficients(ctx.cube_exec, dirty, jones_norm, weights_freq, ctx.xdes, peak_coords,
                                      spectral_coeffs, ws.coeffs_per_chan);

    common::subtract_component(ctx.cube_exec, dirty, conv_psf, ws.coeffs_per_chan, peak_coords, gain);
    common::subtract_component(ctx.exec, scaled_residual, conv2_psf, peak_coords, peak_signed * gain,
                               selected.criterion, threshold);
    // Only the conv2_psf footprint changed: rescan just the touched tiles.
    const auto next = ws.tiled.run_incremental(scaled_residual, peak_coords.row, peak_coords.col, conv2_psf.extent(0),
                                               conv2_psf.extent(1));
    peak_value = next.value;
    peak_index = next.index;
    peak_signed = next.signed_value;

    n_clean_iter++;
  }
  // Callers read dirty, so cube_exec's subtracts must be done.
  ctx.exec.wait();
  ctx.cube_exec.wait();
  FD_PROFILE_MARK("clean_loop synced");
  return n_clean_iter;
}

// The minor-cycle loop on staged inputs; appends this run's components to the context's history.
ddmsc_result run_ddmsc_cycles(context& ctx, const params& p, core::span3d<float> dirty,
                              core::span3d<const float> jones_norm, core::span1d<const float> weights_freq)
{
  FD_PROFILE_FN();

  const std::int64_t psf_nrow = ctx.raw_psfs.extent(2);
  const std::int64_t psf_ncol = ctx.raw_psfs.extent(3);

  const std::int64_t n_freq = dirty.extent(0);
  const std::int64_t n_facets = ctx.raw_psfs.extent(0);
  const std::int64_t dirty_nrows = dirty.extent(1);
  const std::int64_t dirty_ncols = dirty.extent(2);
  const std::int64_t n_order = ctx.xdes.extent(1);
  const int n_scales = static_cast<int>(ctx.scale_sigmas.size());

  FD_LOG_INFO("{}", format_run_banner(p, dirty_nrows, dirty_ncols, n_freq, n_facets, n_scales, psf_nrow, psf_ncol));
  FD_PROFILE_APPINFO(format_run_banner(p, dirty_nrows, dirty_ncols, n_freq, n_facets, n_scales, psf_nrow, psf_ncol));

  // Matches DDFacet's NBand check.
  if (n_freq < 2 * n_order)
    FD_LOG_WARN("run_ddmsc: spectral fit is under-constrained (n_freq={} n_order={}); want n_freq >= 2*n_order.",
                n_freq, n_order);

  run_workspace ws{ctx, p, dirty_nrows, dirty_ncols, n_freq, weights_freq};

  FD_PROFILE_MARK("init/mean_residual");
  const core::span2d<float> mean_residual(ws.mean_residual_buf);
  linalg::weighted_sum(ctx.exec, dirty, weights_freq, mean_residual);

  auto [track_flux, track_rms] = matrix::max_and_rms(ctx.exec, mean_residual, ctx.mask, p.clean_negative);

  // A previous major cycle overflowed.
  if (!std::isfinite(track_flux) || !std::isfinite(track_rms))
    throw std::invalid_argument(
        fmt::format("run_ddmsc: input residual is not finite (peak={}, rms={})", track_flux, track_rms));

  const stop_limits limits = compute_stop_flux(p, track_flux, track_rms);
  const float stop_flux = limits.stop_flux;

  FD_LOG_INFO(
      "run_ddmsc: initial peak_flux={:.8f} rms={:.8f} stop_flux={:.8f} "
      "(rms_lim={:.8f} peak_lim={:.8f} sidelobe_lim={:.8f} floor={:.8f})",
      track_flux, track_rms, stop_flux, limits.rms, limits.peak, limits.sidelobe, p.flux_threshold);

  common::convergence deconv_convergence{p.max_iteration, stop_flux, 5, p.divergence_factor,
                                         common::scale_stall_tracker{n_scales, 5, p.scale_stall_threshold}};
  deconv_convergence.init(track_flux, track_rms);

  ddmsc_result result{core::narrow<std::size_t>(std::int64_t{p.max_iteration} + p.max_clean_iteration)};
  result.initial_flux = track_flux;
  result.initial_rms = track_rms;
  int last_selected_scale = -1;
  // Switched to the auto-mask once, when it engages.
  core::span3d<const bool> scale_masks(ctx.mask.data_handle(), 1, ctx.mask.extent(0), ctx.mask.extent(1));
  core::span2d<const bool> stats_mask = ctx.mask;

  ctx.exec.wait();
  while (!deconv_convergence.should_stop()) {
    FD_PROFILE_FRAME();
    FD_PROFILE_SCOPE("outer_iter");
    result.outer_iterations++;
    FD_LOG_DEBUG("run_ddmsc: outer iter start total_iterations={} track_flux={:.8f} track_rms={:.8f}",
                 deconv_convergence.iteration(), track_flux, track_rms);

    const std::optional<float> auto_mask_threshold = auto_mask_trigger(p, track_flux, track_rms);
    if (auto_mask_threshold && !ws.mask_per_scale) {
      FD_LOG_INFO("Start auto masking at threshold {}", *auto_mask_threshold);
      ws.mask_per_scale = build_scale_mask(ctx, result, weights_freq, dirty_nrows, dirty_ncols);
      ws.merged_active_mask = ctx.exec.alloc_mdcontainer<bool>(dirty_nrows, dirty_ncols);
      scale_masks = *ws.mask_per_scale;
      // A peak no scale can reach must not hold the loop above the stop flux.
      stats_mask = *ws.merged_active_mask;
      deconv_convergence.reset_stalls();
    }

    const auto retired = deconv_convergence.get_all_stalled();
    if (ws.mask_per_scale) {
      std::vector<int> active;
      for (int s = 0; s < n_scales; s++)
        if (std::ranges::find(retired, s) == retired.end()) active.push_back(s);
      common::combine_scale_masks(ctx.exec, *ws.mask_per_scale, active, *ws.merged_active_mask);
    }
    const scale::scale_result selected =
        scale::select_best_scale(ctx.exec, ws.scale_conv, mean_residual, ctx.scale_sigmas, ctx.scale_bias, retired,
                                 scale_masks, p.clean_negative);
    const int selected_scale_idx = selected.scale;

    const int n_clean_iter =
        clean_loop(ws, ctx, p, selected, dirty, jones_norm, weights_freq, deconv_convergence.iteration(), result);
    FD_PROFILE_PLOT("clean_iters", static_cast<std::int64_t>(n_clean_iter));

    // The residual is untouched when no component was found, so the previous stats still hold.
    if (n_clean_iter == 0) {
      FD_LOG_INFO("run_ddmsc: no components found, stopping");
      deconv_convergence.track(track_flux, track_rms, 0, selected_scale_idx);
      continue;
    }

    linalg::weighted_sum(ctx.exec, dirty, weights_freq, mean_residual);

    auto [this_flux, this_rms] = matrix::max_and_rms(ctx.exec, mean_residual, stats_mask, p.clean_negative);

    deconv_convergence.track(this_flux, this_rms, n_clean_iter, selected_scale_idx);

    if (last_selected_scale != selected_scale_idx) {
      const float flux_to_go = this_flux - stop_flux;
      FD_LOG_INFO("run_ddmsc: [iter={}] scale={} peak_flux={:.8f} rms={:.8f} flux_to_go={:.8f}",
                  deconv_convergence.iteration(), selected_scale_idx, this_flux, this_rms, flux_to_go);
      last_selected_scale = selected_scale_idx;
    }

    FD_LOG_DEBUG("run_ddmsc: outer iter end delta_flux={:+.8f} delta_rms={:+.8f}", this_flux - track_flux,
                 this_rms - track_rms);

    track_flux = this_flux;
    track_rms = this_rms;

    if (deconv_convergence.is_stall(selected_scale_idx))
      FD_LOG_INFO("run_ddmsc: retired scale {} due to stall", selected_scale_idx);
  }

  // Closes the last iteration's frame; the loop body only opens a new one.
  FD_PROFILE_FRAME();

  FD_PROFILE_MARK("finalize");

  const int total_iterations = deconv_convergence.iteration();
  result.set_coeffs(ctx.cube_exec, core::span2d<const float>{ws.all_coeffs.data_handle(), total_iterations, n_order});
  const auto final_stats = matrix::max_and_rms(ctx.exec, mean_residual, ctx.mask, p.clean_negative);
  result.final_flux = final_stats.max;
  result.final_rms = final_stats.rms;
  result.stop_flux = stop_flux;
  result.total_iterations = total_iterations;
  result.status = deconv_convergence.status();

  for (const auto& [r, c] : result.peak_coords) ctx.historical_peak_coords.push_back({r, c});
  ctx.historical_scales.insert(ctx.historical_scales.end(), result.scales.begin(), result.scales.end());

  return result;
}

}  // namespace

Ddmsc::Ddmsc(core::host_span4d<const float> raw_psfs, core::host_span2d<const float> xdes,
             core::host_span2d<const bool> mask, core::host_span1d<const float> scale_sigmas,
             core::host_span1d<const float> scale_bias, core::host_span2d<const int> map_pixel_facet,
             std::int64_t dirty_nrow, std::int64_t dirty_ncol, int exec_device)
    : ctx(exec_device, raw_psfs, xdes, mask, scale_sigmas, scale_bias, map_pixel_facet, dirty_nrow, dirty_ncol)
{
}

ddmsc_result Ddmsc::run(core::host_span3d<float> dirty, core::host_span3d<const float> jones_norm,
                        core::host_span1d<const float> weights_freq)
{
  const auto t0 = std::chrono::steady_clock::now();
  auto staged_dirty = ctx.exec.stage(dirty);
  auto staged_jones = ctx.exec.stage(jones_norm);
  auto staged_weights = ctx.exec.stage(weights_freq);

  ddmsc_result result = run_ddmsc_cycles(ctx, cfg, staged_dirty, staged_jones, staged_weights);

  ctx.exec.unstage(staged_dirty, dirty.data_handle());
  ctx.exec.wait();

  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  constexpr double mib = 1 << 20;
  FD_LOG_INFO(
      "run_ddmsc: completed in {:.3f} ms\n"
      "  result | components={}  outer={}  exit={}\n"
      "  flux   | {:.6g} -> {:.6g}  stop={:.6g}\n"
      "  rms    | {:.6g} -> {:.6g}\n"
      "  pool   | used={:.1f} MiB  reserved={:.1f} MiB  peak={:.1f} MiB",
      ms, result.n_components(), result.outer_iterations, common::to_string(result.status), result.initial_flux,
      result.final_flux, result.stop_flux, result.initial_rms, result.final_rms,
      static_cast<double>(ctx.resources.pool_used_bytes()) / mib,
      static_cast<double>(ctx.resources.pool_reserved_bytes()) / mib,
      static_cast<double>(ctx.resources.pool_peak_used_bytes()) / mib);
  return result;
}

}  // namespace fast_deconv::algorithm::ddmsc
