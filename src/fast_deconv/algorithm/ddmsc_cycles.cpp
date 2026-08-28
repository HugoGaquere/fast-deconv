#include <algorithm>
#include <cmath>
#include <emu/submdspan.hpp>
#include <fast_deconv/algorithm/ddmsc_cycles.hpp>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/common/clean.hpp>
#include <fast_deconv/common/convergence.hpp>
#include <fast_deconv/common/gain.hpp>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/common/multi_frequency.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/core/nvtx.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/matrix/stats.hpp>
#include <fast_deconv/matrix/tiled_argmax.hpp>
#include <fast_deconv/util/utils.hpp>
#include <stdexcept>

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

}  // namespace

ddmsc_result run_ddmsc_cycles(context& ctx, const params& p, core::span3d<float>& dirty,
                              const core::span3d<const float>& jones_norm,
                              const core::span1d<const float>& weights_freq)
{
  FD_NVTX_RANGE_FN();
  // log::set_level(spdlog::level::debug);  // disabled for benchmarking

  // First call builds the device state: pool, streams, staged inputs, FFT plans.
  auto& device = ctx.state();
  auto& scale_ctx = device.scale_convolve;
  auto& psf_ctx = device.psf_convolve;
  const linalg::fft_dims& scale_dims = scale_ctx.dims();
  const linalg::fft_dims& psf_dims = psf_ctx.dims();

  // stream_a is the context's compute stream — the FFT plans are bound to it,
  // so the convolutions and the surrounding kernels share one stream.
  // stream_b is the context's aux stream for the clean-loop per-channel
  // fit/subtract path; everything it touches (dirty, the coeff buffers) is
  // allocated and driven on it.
  const auto& stream_a = device.compute_stream;
  const auto& stream_b = device.aux_stream;

  const uint32_t n_freq = dirty.extent(0);
  const uint32_t n_facets = device.raw_psfs_d.extent(0);
  const size_t dirty_nrows = dirty.extent(1);
  const size_t dirty_ncols = dirty.extent(2);
  const size_t dirty_npix = dirty_nrows * dirty_ncols;
  const size_t mean_residual_n_items = dirty_npix;
  const int n_order = device.xdes_d.extent(1);
  const int n_scales = static_cast<int>(device.scale_sigmas_d.size());

  FD_LOG_INFO("{}", format_run_banner(p, dirty_nrows, dirty_ncols, n_freq, n_facets, n_scales, psf_dims.input_nrow,
                                      psf_dims.input_ncol));

  // The spectral fit needs about two bands per coefficient to be a fit rather than an
  // interpolation; at or below one, it stops being identifiable and only the minimum-norm
  // regulariser keeps the coefficients bounded. Matches DDFacet's own NBand check.
  if (static_cast<int>(n_freq) < 2 * n_order)
    FD_LOG_WARN(
        "run_ddmsc: spectral fit is under-constrained (n_freq={} n_order={}); want n_freq >= 2*n_order. "
        "Coefficients are extrapolated to the degrid frequencies, where the unconstrained directions dominate.",
        n_freq, n_order);

  FD_NVTX_MARK("init/queue_residual_and_kernels");
  // Initial mean residual: owning buffer plus a mutable view that gets
  // re-seated onto scale slices during the loop.
  auto mean_residual_buf = stream_a.alloc_mdcontainer_async<float>(dirty_nrows, dirty_ncols);
  core::span2d<float> mean_residual(mean_residual_buf.data_handle(), dirty_nrows, dirty_ncols);
  linalg::weighted_sum_async(stream_a, dirty, weights_freq, mean_residual);

  FD_NVTX_MARK("init/initial_stats begin");
  // Fused max + rms reduction: one CUB sweep, one D2H, one sync per call.
  matrix::stats_ctx stats_ws{stream_a, mean_residual_n_items, p.clean_negative};

  // Compute and track initial flux and RMS
  auto [track_flux, track_rms] = stats_ws.run(mean_residual, device.mask_d);
  FD_NVTX_MARK("init/initial_stats end");

  // Overflowed in a previous cycle
  if (!std::isfinite(track_flux) || !std::isfinite(track_rms))
    throw std::invalid_argument(
        fmt::format("run_ddmsc: input residual is not finite (peak={}, rms={})", track_flux, track_rms));

  // Compose the stop-flux threshold from the four limits.
  const float fluxlimit_rms = p.stop_rms_factor * track_rms;
  const float fluxlimit_peak = p.stop_peak_factor * track_flux;
  const float sidelobe_coeff =
      (p.stop_cycle_factor != 0.0f)
          ? ((p.stop_cycle_factor - 1.0f) / 4.0f * (1.0f - p.stop_sidelobe_level) + p.stop_sidelobe_level)
          : 0.0f;
  const float fluxlimit_sidelobe = sidelobe_coeff * track_flux;
  const float stop_flux = std::max({p.flux_threshold, fluxlimit_rms, fluxlimit_peak, fluxlimit_sidelobe});

  FD_LOG_INFO(
      "run_ddmsc: initial pak_flux={:.8f} rms={:.8f} stop_flux={:.8f} "
      "(rms_lim={:.8f} peak_lim={:.8f} sidelobe_lim={:.8f} floor={:.8f})",
      track_flux, track_rms, stop_flux, fluxlimit_rms, fluxlimit_peak, fluxlimit_sidelobe, p.flux_threshold);

  // Init convergence and scales watchers
  common::convergence deconv_convergence{p.max_iteration, stop_flux, 5, p.divergence_factor,
                                         common::scale_stall_tracker{n_scales, 5, p.scale_stall_threshold}};

  deconv_convergence.init(track_flux, track_rms);

  // Shared device buffer for all component coefficients across all outer
  // iterations. Written by fit_coefficients on stream_b, so it lives on
  // stream_b's timeline (alloc-follows-use).
  const std::size_t coeffs_capacity = static_cast<std::size_t>(p.max_iteration + p.max_clean_iteration) * n_order;
  auto all_coeffs = stream_b.alloc_mdcontainer_async<float>(coeffs_capacity);

  // Per-channel coefficient scratch for the clean loop: written and consumed
  // only on stream_b, fixed size, so allocated once for the whole call.
  auto coeffs_per_chan = stream_b.alloc_mdcontainer_async<float>(n_freq);

  // Prepared argmax for repeated peak lookups over mean_residual
  matrix::argmax_ctx peak_ws{stream_a, mean_residual_n_items};

  // Tiled argmax for the clean loop: after a clean subtraction only the
  // conv2_psf footprint is dirtied, so we recompute just the touched tiles instead
  // of rescanning all of mean_residual. Re-seeded with a full pass each outer iter.
  matrix::tiled_argmax_ctx tiled_ws{stream_a, mean_residual.extents(), 64};

  FD_NVTX_MARK("init/precompute_psfs begin");
  // auto all_conv_psfs =
  //     stream_a.alloc_mdcontainer_async<float>(n_scales, n_facets, n_freq, psf_dims.input_nrow, psf_dims.input_ncol);
  // auto all_conv2_psfs =
  //     stream_a.alloc_mdcontainer_async<float>(n_scales, n_facets, psf_dims.input_nrow, psf_dims.input_ncol);
  // scale::convolve_psfs_with_scales_async(psf_ctx, device.raw_psfs_d, device.scale_sigmas_d, weights_freq,
  // all_conv_psfs,
  //                                        all_conv2_psfs);

  // TEMP: test no conv psf precomputation
  auto conv_psfs = stream_a.alloc_mdcontainer_async<float>(n_facets, n_freq, psf_dims.input_nrow, psf_dims.input_ncol);
  auto conv2_psfs = stream_a.alloc_mdcontainer_async<float>(n_facets, psf_dims.input_nrow, psf_dims.input_ncol);
  FD_NVTX_MARK("init/precompute_psfs end");
  FD_NVTX_MARK("init/compute_gains begin");
  // auto all_gains = common::compute_all_gains_batched(stream_a, all_conv_psfs, weights_freq, p.gamma);
  FD_NVTX_MARK("init/compute_gains end");

  // Loop-only buffers
  auto scale_kernels = stream_a.alloc_mdcontainer_async<float>(n_scales, scale_dims.freq_nrow, scale_dims.freq_ncol);
  scale::make_gaussian_kernels_async(stream_a, device.scale_sigmas_d, scale_dims.padded_ncol, scale_kernels);

  FD_LOG_DEBUG("run_ddmsc: built {} scale kernels in freq domain ({}x{}, {} floats total)", n_scales,
               scale_dims.freq_nrow, scale_dims.freq_ncol, scale_kernels.size());

  auto scales_x_dirty = stream_a.alloc_mdcontainer_async<float>(n_scales, dirty_nrows, dirty_ncols);

  // TODO: fix that
  int total_iterations = 0;
  ddmsc_result result{p.max_iteration, n_order};

  // We dont allocate memory for mask_per_scale while we didnt trigger the auto masking
  core::cont3d<bool> mask_per_scale;
  bool is_auto_mask_initialized = false;

  int last_selected_scale = -1;

  // Convolved PSFs are recomputed only when the selected scale changes.
  std::vector<float> gains;
  int cached_scale = -1;

  stream_a.wait();
  // Loop over scales
  while (!deconv_convergence.should_stop()) {
    FD_NVTX_RANGE("outer_iter");
    FD_LOG_DEBUG("run_ddmsc: outer iter start total_iterations={} track_flux={:.8f} track_rms={:.8f}", total_iterations,
                 track_flux, track_rms);

    float auto_mask_threshold =
        p.auto_mask_peak_threshold.value_or(p.auto_mask_rms_threshold.value_or(0.f) * track_rms);

    bool activate_auto_mask = (p.enable_auto_mask && track_flux <= auto_mask_threshold) | p.force_enable_auto_mask;

    if (activate_auto_mask && !is_auto_mask_initialized) {
      FD_NVTX_RANGE("build_auto_mask");
      FD_LOG_INFO("Start auto masking at threshold {}", auto_mask_threshold);
      mask_per_scale = stream_a.alloc_mdcontainer_async<bool>(n_scales, dirty_nrows, dirty_ncols);

      const int central_facet_idx = ctx.map_pixel_facet(dirty_nrows / 2, dirty_ncols / 2);
      core::span3d<float> central_facet_psfs = emu::submdspan(device.raw_psfs_d, central_facet_idx);

      // Merge history-from-previous-calls with components found so far in this call.
      std::vector<std::pair<int, int>> all_coords = ctx.historical_peak_coords;
      all_coords.insert(all_coords.end(), result.peak_coords.begin(), result.peak_coords.end());
      std::vector<int> all_scales = ctx.historical_scales;
      all_scales.insert(all_scales.end(), result.scales.begin(), result.scales.end());

      const float fft_padding = static_cast<float>(psf_dims.padded_nrow) / psf_dims.input_nrow;
      common::build_auto_mask(stream_a, all_coords, all_scales, central_facet_psfs, weights_freq, device.scale_sigmas_d,
                              fft_padding, device.mask_d, mask_per_scale);

      is_auto_mask_initialized = true;
    }

    FD_NVTX_MARK("convolve_with_scales");
    scale::convolve_with_scales(scale_ctx, mean_residual, scale_kernels, scales_x_dirty);

    FD_NVTX_MARK("mask_and_abs");
    if (activate_auto_mask)
      common::mask_and_abs_async(stream_a, scales_x_dirty, mask_per_scale, -std::numeric_limits<float>::infinity(),
                                 p.clean_negative);
    else
      common::mask_and_abs_async(stream_a, scales_x_dirty, device.mask_d, -std::numeric_limits<float>::infinity(),
                                 p.clean_negative);

    FD_NVTX_MARK("scale_selection");
    int selected_scale_idx =
        scale::scale_selection(stream_a, scales_x_dirty, ctx.scale_bias, deconv_convergence.get_all_stalled());

    // FD_LOG_INFO("run_ddmsc: selected scale {}, auto_mask {}", selected_scale_idx, activate_auto_mask);

    mean_residual = emu::submdspan(scales_x_dirty, selected_scale_idx);

    // Filled on stream_a; the argmax/gain host syncs below retire it before stream_b reads it.
    if (selected_scale_idx != cached_scale) {
      core::span1d<float> sigma_view(device.scale_sigmas_d.data_handle() + selected_scale_idx, 1);
      scale::convolve_psfs_with_scale_async(psf_ctx, device.raw_psfs_d, sigma_view, selected_scale_idx, weights_freq,
                                            conv_psfs, conv2_psfs);
      // Scale 0 is the identity kernel, so gain == gamma (as compute_all_gains_batched does).
      gains = (selected_scale_idx == 0) ? std::vector<float>(n_facets, p.gamma)
                                        : common::compute_gain_batched(stream_a, conv_psfs, weights_freq, p.gamma);
      cached_scale = selected_scale_idx;
    }

    auto [peak_value, peak_index] = peak_ws.run(mean_residual);

    const float threshold = peak_value * p.peak_factor;

    common::mask_less_than_threshold(stream_a, mean_residual, threshold, -std::numeric_limits<float>::infinity());

    // Seed the tile cache against the masked buffer for this scale. The peak is
    // unchanged by masking (it only removes sub-threshold values), so we keep the
    // peak_value/peak_index from the argmax above and use this purely to seed.
    tiled_ws.run(mean_residual);

    FD_LOG_DEBUG("run_ddmsc: clean loop start scale={} peak={:.8f} threshold={:.8f} max_clean_iter={}",
                 selected_scale_idx, peak_value, threshold, p.max_clean_iteration);

    FD_NVTX_MARK("clean_loop begin");
    // Clean loop over mean residual
    int n_clean_iter = 0;
    while (peak_value > threshold && n_clean_iter < p.max_clean_iteration) {
      FD_NVTX_RANGE("minor_iter");
      const auto peak_coords = util::unravel_index_2D(peak_index, dirty_ncols);
      const int facet_idx = ctx.map_pixel_facet(peak_coords.first, peak_coords.second);
      // const float gain = all_gains.at(gain_offset + facet_idx);
      const float gain = gains.at(facet_idx);

      result.add_component(peak_coords, selected_scale_idx, gain);

      FD_LOG_DEBUG("run_ddmsc:   [sub={}] peak={:.8f} at ({},{}) facet={} gain={:.6f}", n_clean_iter, peak_value,
                   peak_coords.first, peak_coords.second, facet_idx, gain);

      core::span3d<float> conv_psf = emu::submdspan(conv_psfs, facet_idx);
      core::span2d<float> conv2_psf = emu::submdspan(conv2_psfs, facet_idx);

      const std::size_t coeffs_offset = (total_iterations + n_clean_iter) * n_order;
      auto spectral_coeffs = core::span1d<float>(all_coeffs.data_handle() + coeffs_offset, n_order);
      multi_frequency::fit_coefficients(stream_b, dirty, jones_norm, weights_freq, device.xdes_d, peak_coords,
                                        spectral_coeffs, coeffs_per_chan);

      common::subtract_component_async(stream_b, dirty, conv_psf, coeffs_per_chan, peak_coords, gain);
      common::subtract_component_async(stream_a, mean_residual, conv2_psf, peak_coords, peak_value * gain);
      // Only the conv2_psf footprint centered on peak_coords was dirtied; refresh
      // just the touched tiles and re-combine against the cached ones.
      const auto next = tiled_ws.run_incremental(mean_residual, peak_coords.first, peak_coords.second,
                                                 conv2_psf.extent(0), conv2_psf.extent(1));
      peak_value = next.value;
      peak_index = next.index;

      n_clean_iter++;
    }

    stream_a.wait();
    // stream_b must finish its per-channel subtracts on `dirty` before
    // stream_a reads it in the weighted_sum below.
    stream_b.wait();
    FD_NVTX_MARK("clean_loop end");

    // FD_LOG_INFO("run_ddmsc: scale {} produced {} clean iterations", selected_scale_idx, n_clean_iter);

    // The residual is untouched when no component was found, so the previous stats still hold.
    if (n_clean_iter == 0) {
      FD_LOG_INFO("ddmsc_minor_cycles: no components found, stopping");
      deconv_convergence.track(track_flux, track_rms, 0, selected_scale_idx);
      continue;
    }

    total_iterations += n_clean_iter;

    FD_NVTX_MARK("post_iter_stats begin");
    // Reset mean_residual view to the original owning buffer to store the new mean
    mean_residual = core::span2d<float>(mean_residual_buf.data_handle(), dirty_nrows, dirty_ncols);
    linalg::weighted_sum_async(stream_a, dirty, weights_freq, mean_residual);

    // TODO(guards): stats use device.mask_d while the sub-minor loop searches mask_per_scale when the
    // auto-mask is active, so a bright pixel outside the auto-mask is never cleanable yet still
    // sets track_flux and the stop_flux comparison (observed on a LOFAR run with AutoMask=True:
    // peak_flux pinned at 0.03776400 for ~1700 iterations while rms kept falling). Consider
    // computing the stats against the same mask the peak search is allowed to clean.
    auto [this_flux, this_rms] = stats_ws.run(mean_residual, device.mask_d);
    FD_NVTX_MARK("post_iter_stats end");

    if (last_selected_scale != selected_scale_idx) {
      const float flux_to_go = this_flux - stop_flux;
      FD_LOG_INFO("run_ddmsc: [iter={}] scale={} peak_flux={:.8f} rms={:.8f} flux_to_go={:.8f}", total_iterations,
                  selected_scale_idx, this_flux, this_rms, flux_to_go);
      last_selected_scale = selected_scale_idx;
    }

    deconv_convergence.track(this_flux, this_rms, n_clean_iter, selected_scale_idx);

    FD_LOG_DEBUG("run_ddmsc: outer iter end delta_flux={:+.8f} delta_rms={:+.8f}", this_flux - track_flux,
                 this_rms - track_rms);

    track_flux = this_flux;
    track_rms = this_rms;

    if (deconv_convergence.is_stall(selected_scale_idx))
      FD_LOG_INFO("ddmsc_minor_cycles: retired scale {} due to stall", selected_scale_idx);
  }

  FD_NVTX_MARK("finalize begin");
  stream_a.wait();
  stream_b.wait();

  FD_LOG_INFO("run_ddmsc: completed ({} iterations, exit={})", deconv_convergence.iteration(),
              common::to_string(deconv_convergence.status()));

  result.add_coeffs_from_device(stream_b, core::span2d<float>{all_coeffs.data_handle(), total_iterations, n_order});
  result.final_flux = track_flux;
  result.stop_flux = stop_flux;
  result.total_iterations = total_iterations;
  result.status = deconv_convergence.status();

  ctx.historical_peak_coords.insert(ctx.historical_peak_coords.end(), result.peak_coords.begin(),
                                    result.peak_coords.end());
  ctx.historical_scales.insert(ctx.historical_scales.end(), result.scales.begin(), result.scales.end());

  return result;
}

}  // namespace fast_deconv::algorithm::ddmsc
