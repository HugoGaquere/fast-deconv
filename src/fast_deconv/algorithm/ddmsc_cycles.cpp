#include <algorithm>
#include <cmath>
#include <cstdint>
#include <emu/submdspan.hpp>
#include <fast_deconv/algorithm/ddmsc_cycles.hpp>
#include <fast_deconv/algorithm/psf_convolution.hpp>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/common/clean.hpp>
#include <fast_deconv/common/convergence.hpp>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/common/multi_frequency.hpp>
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

std::string format_run_banner(const params& p, std::size_t dirty_nrows, std::size_t dirty_ncols, uint32_t n_freq,
                              uint32_t n_facets, int n_scales, int psf_nrow, int psf_ncol)
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
  if ((p.enable_auto_mask && flux <= threshold) || p.force_enable_auto_mask) return threshold;
  return std::nullopt;
}

struct run_workspace {
  core::cont2d<float> mean_residual_buf;
  matrix::stats_ctx stats;
  matrix::tiled_argmax_ctx tiled;
  const linalg::gaussian_convolution_ctx scale_conv;
  psf_convolution psf_cache;
  int cached_scale = -1;  // scale psf_cache holds
  // Coefficient buffers live on aux_stream, which writes them.
  core::cont1d<float> all_coeffs;
  core::cont1d<float> coeffs_per_chan;
  // Built once, the first time the auto-mask engages.
  std::optional<core::cont3d<bool>> mask_per_scale;

  run_workspace(const context& ctx, const params& p, std::size_t nrow, std::size_t ncol, uint32_t n_freq,
                const core::span1d<const float>& weights_freq)
      : mean_residual_buf(ctx.compute_stream.alloc_mdcontainer_async<float>(nrow, ncol)),
        stats(ctx.compute_stream, nrow * ncol, p.clean_negative),
        tiled(ctx.compute_stream, mean_residual_buf.extents(), 64),
        scale_conv(ctx.compute_stream, 1, ctx.dirty_nrow, ctx.dirty_ncol, linalg::max_gaussian_reach(ctx.scale_sigmas)),
        psf_cache(ctx.compute_stream, ctx.raw_psfs, ctx.scale_sigmas, weights_freq, p.gamma),
        all_coeffs(ctx.aux_stream.alloc_mdcontainer_async<float>(
            static_cast<std::size_t>(p.max_iteration + p.max_clean_iteration) * ctx.xdes.extent(1))),
        coeffs_per_chan(ctx.aux_stream.alloc_mdcontainer_async<float>(n_freq))
  {
  }
};

/// Per-scale masks around every component so far, previous calls included.
core::cont3d<bool> build_scale_mask(const context& ctx, const ddmsc_result& result,
                                    const core::span1d<const float>& weights_freq, std::size_t nrow, std::size_t ncol)
{
  FD_PROFILE_FN();
  const auto& stream_a = ctx.compute_stream;
  auto mask_per_scale = stream_a.alloc_mdcontainer_async<bool>(ctx.scale_sigmas.size(), nrow, ncol);

  const int central_facet_idx = ctx.map_pixel_facet(nrow / 2, ncol / 2);
  core::span3d<float> central_facet_psfs = emu::submdspan(ctx.raw_psfs, central_facet_idx);

  std::vector<common::index2d> all_coords = ctx.historical_peak_coords;
  for (const auto& [r, c] : result.peak_coords) all_coords.push_back({r, c});
  std::vector<int> all_scales = ctx.historical_scales;
  all_scales.insert(all_scales.end(), result.scales.begin(), result.scales.end());

  common::build_auto_mask(stream_a, all_coords, all_scales, central_facet_psfs, weights_freq, ctx.scale_sigmas,
                          ctx.mask, mask_per_scale);
  return mask_per_scale;
}

void select_psf_scale(run_workspace& ws, const params& p, int scale)
{
  FD_PROFILE_FN();
  // One scale stays resident; stream_b, which reads the entries, was drained last iteration.
  if (scale != ws.cached_scale && p.psf_cache_policy != psf_cache_mode::eager_all) {
    ws.psf_cache.clear();
    ws.cached_scale = scale;
  }

  // Other policies fill misses per facet in get().
  if (p.psf_cache_policy == psf_cache_mode::lazy_scale) ws.psf_cache.prefetch_scale(scale);
}

/// Returns the number of components found.
int clean_loop(run_workspace& ws, const context& ctx, const params& p, const scale::scale_result& selected,
               core::span3d<float>& dirty, const core::span3d<const float>& jones_norm,
               const core::span1d<const float>& weights_freq, int first_component, ddmsc_result& result)
{
  FD_PROFILE_FN();
  const auto& stream_a = ctx.compute_stream;
  const auto& stream_b = ctx.aux_stream;
  const std::size_t dirty_ncols = dirty.extent(2);
  const int n_order = ctx.xdes.extent(1);
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
    const psf_convolution::entry psf = ws.psf_cache.get(selected_scale_idx, facet_idx);
    const float gain = psf.gain;

    result.add_component(peak_coords, selected_scale_idx, gain);

    FD_LOG_DEBUG("run_ddmsc:   [sub={}] peak={:.8f} at ({},{}) facet={} gain={:.6f}", n_clean_iter, peak_value,
                 peak_coords.row, peak_coords.col, facet_idx, gain);

    core::span3d<const float> conv_psf = psf.conv;
    core::span2d<const float> conv2_psf = psf.conv2;

    const std::size_t coeffs_offset = static_cast<std::size_t>(first_component + n_clean_iter) * n_order;
    auto spectral_coeffs = core::span1d<float>(ws.all_coeffs.data_handle() + coeffs_offset, n_order);
    multi_frequency::fit_coefficients(stream_b, dirty, jones_norm, weights_freq, ctx.xdes, peak_coords, spectral_coeffs,
                                      ws.coeffs_per_chan);

    common::subtract_component_async(stream_b, dirty, conv_psf, ws.coeffs_per_chan, peak_coords, gain);
    common::subtract_component_async(stream_a, scaled_residual, conv2_psf, peak_coords, peak_signed * gain,
                                     selected.criterion, threshold);
    // Only the conv2_psf footprint changed: rescan just the touched tiles.
    const auto next = ws.tiled.run_incremental(scaled_residual, peak_coords.row, peak_coords.col, conv2_psf.extent(0),
                                               conv2_psf.extent(1));
    peak_value = next.value;
    peak_index = next.index;
    peak_signed = next.signed_value;

    n_clean_iter++;
  }
  return n_clean_iter;
}

}  // namespace

ddmsc_result run_ddmsc_cycles(context& ctx, const params& p, core::span3d<float>& dirty,
                              const core::span3d<const float>& jones_norm,
                              const core::span1d<const float>& weights_freq)
{
  FD_PROFILE_FN();
  // log::set_level(spdlog::level::debug);  // disabled for benchmarking

  const int psf_nrow = ctx.raw_psfs.extent(2);
  const int psf_ncol = ctx.raw_psfs.extent(3);

  // stream_b runs the per-channel fit/subtract on dirty, overlapping stream_a.
  const auto& stream_a = ctx.compute_stream;
  const auto& stream_b = ctx.aux_stream;

  const uint32_t n_freq = dirty.extent(0);
  const uint32_t n_facets = ctx.raw_psfs.extent(0);
  const size_t dirty_nrows = dirty.extent(1);
  const size_t dirty_ncols = dirty.extent(2);
  const int n_order = ctx.xdes.extent(1);
  const int n_scales = static_cast<int>(ctx.scale_sigmas.size());

  FD_LOG_INFO("{}", format_run_banner(p, dirty_nrows, dirty_ncols, n_freq, n_facets, n_scales, psf_nrow, psf_ncol));
  // Also in the trace, so a .tracy file says what it ran on.
  FD_PROFILE_APPINFO(format_run_banner(p, dirty_nrows, dirty_ncols, n_freq, n_facets, n_scales, psf_nrow, psf_ncol));

  // Matches DDFacet's NBand check.
  if (static_cast<int>(n_freq) < 2 * n_order)
    FD_LOG_WARN(
        "run_ddmsc: spectral fit is under-constrained (n_freq={} n_order={}); want n_freq >= 2*n_order. "
        "Coefficients are extrapolated to the degrid frequencies, where the unconstrained directions dominate.",
        n_freq, n_order);

  run_workspace ws{ctx, p, dirty_nrows, dirty_ncols, n_freq, weights_freq};

  FD_PROFILE_MARK("init/mean_residual");
  const core::span2d<float> mean_residual(ws.mean_residual_buf);
  linalg::weighted_sum_async(stream_a, dirty, weights_freq, mean_residual);

  auto [track_flux, track_rms] = ws.stats.run(mean_residual, ctx.mask);

  // A previous major cycle overflowed.
  if (!std::isfinite(track_flux) || !std::isfinite(track_rms))
    throw std::invalid_argument(
        fmt::format("run_ddmsc: input residual is not finite (peak={}, rms={})", track_flux, track_rms));

  const stop_limits limits = compute_stop_flux(p, track_flux, track_rms);
  const float stop_flux = limits.stop_flux;

  FD_LOG_INFO(
      "run_ddmsc: initial pak_flux={:.8f} rms={:.8f} stop_flux={:.8f} "
      "(rms_lim={:.8f} peak_lim={:.8f} sidelobe_lim={:.8f} floor={:.8f})",
      track_flux, track_rms, stop_flux, limits.rms, limits.peak, limits.sidelobe, p.flux_threshold);

  // iteration() counts the components found.
  common::convergence deconv_convergence{p.max_iteration, stop_flux, 5, p.divergence_factor,
                                         common::scale_stall_tracker{n_scales, 5, p.scale_stall_threshold}};
  deconv_convergence.init(track_flux, track_rms);

  FD_PROFILE_MARK("init/psf_cache");
  if (p.psf_cache_policy == psf_cache_mode::eager_all) ws.psf_cache.prefetch_all();

  ddmsc_result result{static_cast<std::size_t>(p.max_iteration + p.max_clean_iteration)};
  int last_selected_scale = -1;

  stream_a.wait();
  while (!deconv_convergence.should_stop()) {
    FD_PROFILE_FRAME();
    FD_PROFILE_SCOPE("outer_iter");
    FD_LOG_DEBUG("run_ddmsc: outer iter start total_iterations={} track_flux={:.8f} track_rms={:.8f}",
                 deconv_convergence.iteration(), track_flux, track_rms);
    FD_PROFILE_PLOT("peak_flux", track_flux);
    FD_PROFILE_PLOT("rms", track_rms);
    FD_PROFILE_PLOT("pool_used_mib", static_cast<double>(ctx.resources.pool_used_bytes()) / (1 << 20));
    FD_PROFILE_PLOT("pool_reserved_mib", static_cast<double>(ctx.resources.pool_reserved_bytes()) / (1 << 20));
    FD_PROFILE_PLOT("psf_cache_entries", static_cast<std::int64_t>(ws.psf_cache.n_entries()));

    const std::optional<float> auto_mask_threshold = auto_mask_trigger(p, track_flux, track_rms);
    if (auto_mask_threshold && !ws.mask_per_scale) {
      FD_LOG_INFO("Start auto masking at threshold {}", *auto_mask_threshold);
      ws.mask_per_scale = build_scale_mask(ctx, result, weights_freq, dirty_nrows, dirty_ncols);
    }

    const auto retired = deconv_convergence.get_all_stalled();
    const scale::scale_result selected =
        auto_mask_threshold ? scale::select_best_scale(stream_a, ws.scale_conv, mean_residual, ctx.scale_sigmas,
                                                       ctx.scale_bias, retired, *ws.mask_per_scale, p.clean_negative)
                            : scale::select_best_scale(stream_a, ws.scale_conv, mean_residual, ctx.scale_sigmas,
                                                       ctx.scale_bias, retired, ctx.mask, p.clean_negative);
    const int selected_scale_idx = selected.scale;
    select_psf_scale(ws, p, selected_scale_idx);

    const int n_clean_iter =
        clean_loop(ws, ctx, p, selected, dirty, jones_norm, weights_freq, deconv_convergence.iteration(), result);

    stream_a.wait();
    // The weighted_sum below reads dirty, so stream_b's subtracts must be done.
    stream_b.wait();
    FD_PROFILE_MARK("clean_loop synced");
    FD_PROFILE_PLOT("clean_iters", static_cast<std::int64_t>(n_clean_iter));

    // The residual is untouched when no component was found, so the previous stats still hold.
    if (n_clean_iter == 0) {
      FD_LOG_INFO("ddmsc_minor_cycles: no components found, stopping");
      deconv_convergence.track(track_flux, track_rms, 0, selected_scale_idx);
      continue;
    }

    linalg::weighted_sum_async(stream_a, dirty, weights_freq, mean_residual);

    // TODO(guards): stats use ctx.mask while the clean loop may search mask_per_scale.
    auto [this_flux, this_rms] = ws.stats.run(mean_residual, ctx.mask);

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
      FD_LOG_INFO("ddmsc_minor_cycles: retired scale {} due to stall", selected_scale_idx);
  }

  // Closes the last iteration's frame; the loop body only opens a new one.
  FD_PROFILE_FRAME();

  FD_PROFILE_MARK("finalize");
  stream_a.wait();
  stream_b.wait();

  const int total_iterations = deconv_convergence.iteration();
  FD_LOG_INFO("run_ddmsc: completed ({} iterations, exit={})", total_iterations,
              common::to_string(deconv_convergence.status()));

  result.set_coeffs(stream_b, core::span2d<const float>{ws.all_coeffs.data_handle(), total_iterations, n_order});
  result.final_flux = track_flux;
  result.stop_flux = stop_flux;
  result.total_iterations = total_iterations;
  result.status = deconv_convergence.status();

  for (const auto& [r, c] : result.peak_coords) ctx.historical_peak_coords.push_back({r, c});
  ctx.historical_scales.insert(ctx.historical_scales.end(), result.scales.begin(), result.scales.end());

  return result;
}

}  // namespace fast_deconv::algorithm::ddmsc
