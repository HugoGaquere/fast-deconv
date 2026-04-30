#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/algorithm/wscms.hpp>
#include <fast_deconv/common/clean.hpp>
#include <fast_deconv/common/convergence.hpp>
#include <fast_deconv/common/gain.hpp>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/common/multi_frequency.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/matrix/max.hpp>
#include <fast_deconv/matrix/rms.hpp>
#include <fast_deconv/util/utils.hpp>

namespace fast_deconv::algorithm::wscms {

wscms_result run_wscms_cycles(context& ctx, const params& p, core::device_span3d<float>& dirty,
                              const core::device_span3d<float>& jones_norm,
                              const core::device_vect<float>& weights_freq)
{
  log::set_level(spdlog::level::debug);

  auto& exec_resources = ctx.exec_resources;
  auto& ws = ctx.workspace;
  auto& scale_ctx = ws.scale_convolve;
  auto& psf_ctx = ws.psf_convolve;

  const auto& stream_a = exec_resources.get_stream_resources();
  const auto& stream_b = exec_resources.get_stream_resources();

  const uint32_t n_freq = dirty.extent(0);
  const uint32_t n_facets = ws.raw_psfs.extent(0);
  const size_t dirty_nrows = dirty.extent(1);
  const size_t dirty_ncols = dirty.extent(2);
  const size_t dirty_npix = dirty_nrows * dirty_ncols;
  const size_t mean_residual_n_items = dirty_npix;
  const int n_order = ws.xdes.extent(1);
  const int n_scales = static_cast<int>(ws.scale_sigmas.size());

  FD_LOG_INFO(
      "run_wscms: dirty={}x{} n_freq={} n_facets={} n_scales={} psf={}x{} "
      "max_iter={} max_clean_iter={} gamma={:.4f} peak_factor={:.4f} stop_flux={:.6f} "
      "clean_negative={} divergence={:.4f} stall={:.6f}",
      dirty_nrows, dirty_ncols, n_freq, n_facets, n_scales, psf_ctx.input_nrow, psf_ctx.input_ncol, p.max_iteration,
      p.max_clean_iteration, p.gamma, p.peak_factor, p.stop_flux_threshold, p.clean_negative, p.divergence_factor,
      p.scale_stall_threshold);

  // Initial mean residual
  float* mean_residual_ptr = exec_resources.alloc_async<float>(mean_residual_n_items, stream_a);
  core::device_span2d<float> mean_residual(mean_residual_ptr, dirty_nrows, dirty_ncols);
  linalg::weighted_sum_async(stream_a, dirty, weights_freq, mean_residual);

  // Build gaussian kernel
  const uint64_t freq_scales_total = static_cast<int64_t>(scale_ctx.freq_nrow) * scale_ctx.freq_ncol * n_scales;
  float* scale_kernels_ptr = exec_resources.alloc_async<float>(freq_scales_total, stream_a);
  core::device_span3d<float> scale_kernels(scale_kernels_ptr, n_scales, scale_ctx.freq_nrow, scale_ctx.freq_ncol);
  scale::make_gaussian_kernels_async(stream_a, ws.scale_sigmas, scale_ctx.padded_ncol, scale_kernels);

  FD_LOG_DEBUG("run_wscms: built {} scale kernels in freq domain ({}x{}, {} floats total)", n_scales,
               scale_ctx.freq_nrow, scale_ctx.freq_ncol, freq_scales_total);

  // Init convergence and scales watchers
  common::convergence deconv_convergence{p.max_iteration, p.stop_flux_threshold, 5, p.divergence_factor};
  common::scale_stall_tracker scale_stall_tracker{n_scales, 5, p.scale_stall_threshold};

  // Compute and track initial flux and RMS
  float track_flux = matrix::max(exec_resources, stream_a, mean_residual, ws.mask, p.clean_negative);
  float track_rms = matrix::rms(exec_resources, stream_a, mean_residual, ws.mask);

  FD_LOG_INFO("run_wscms: initial peak_flux={:.8f} rms={:.8f}", track_flux, track_rms);

  deconv_convergence.track_flux(track_flux);
  scale_stall_tracker.init_rms(track_rms);

  // Shared device buffer for all component coefficients across all outer iterations.
  const std::size_t coeffs_capacity = static_cast<std::size_t>(p.max_iteration + p.max_clean_iteration) * n_order;
  float* d_all_coeffs = exec_resources.alloc_async<float>(coeffs_capacity, stream_a);

  // Setup workspace for repeated argmax over mean_residual
  matrix::argmax_workspace peak_ws{exec_resources, stream_a, mean_residual_n_items};

  // Allocate memory for the result of each convolution between mean_residual and the scales
  float* scales_x_dirty_ptr = exec_resources.alloc_async<float>(dirty_npix * n_scales, stream_a);
  core::device_span3d<float> scales_x_dirty(scales_x_dirty_ptr, n_scales, dirty_nrows, dirty_ncols);

  // Allocate memory for each psf comvolved with the selected scale
  const int psf_npix = psf_ctx.input_nrow * psf_ctx.input_ncol;
  float* conv_psfs_ptr = exec_resources.alloc_async<float>(n_facets * n_freq * psf_npix, stream_a);
  float* conv2_psfs_ptr = exec_resources.alloc_async<float>(n_facets * psf_npix, stream_a);
  core::device_span4d<float> conv_psfs(conv_psfs_ptr, n_facets, n_freq, psf_ctx.input_nrow, psf_ctx.input_ncol);
  core::device_span3d<float> conv2_psfs(conv2_psfs_ptr, n_facets, psf_ctx.input_nrow, psf_ctx.input_ncol);

  // TODO: fix that
  int total_iterations = 0;
  wscms_result result{p.max_iteration, n_order};

  // We dont allocate memory for mask_per_scale while we didnt trigger the independant masking
  bool* mask_per_scale_ptr = nullptr;
  core::device_span3d<bool> mask_per_scale{mask_per_scale_ptr, n_scales, dirty_nrows, dirty_ncols};
  bool is_independant_mask_initialized = false;

  // Loop over scales
  while (!deconv_convergence.should_stop() && !scale_stall_tracker.all_stalled()) {
    FD_LOG_DEBUG("run_wscms: outer iter start total_iterations={} track_flux={:.8f} track_rms={:.8f}", total_iterations,
                 track_flux, track_rms);

    float scale_dependant_threshold = p.scale_dependant_masking_peak_threshold.value_or(
        p.scale_dependant_masking_rms_threshold.value_or(0.f) * track_rms);

    bool activate_dependant_masking = (p.enable_scale_dependant_masking && track_flux <= scale_dependant_threshold) |
                                      p.force_enable_scale_dependant_masking;

    if (activate_dependant_masking && !is_independant_mask_initialized) {
      FD_LOG_INFO("Start independant scale masking at threshold {}", scale_dependant_threshold);
      mask_per_scale_ptr = stream_a.alloc_async<bool>(n_scales * dirty_nrows * dirty_ncols);
      mask_per_scale = core::device_span3d<bool>{mask_per_scale_ptr, n_scales, dirty_nrows, dirty_ncols};

      const int central_facet_idx = ws.map_pixel_facet(dirty_nrows / 2, dirty_ncols / 2);
      auto central_facet_psfs = core::slice_leading(ws.raw_psfs, central_facet_idx);

      const float fft_padding = static_cast<float>(psf_ctx.padded_nrow) / psf_ctx.input_nrow;
      common::build_independant_scale_mask(exec_resources, stream_a, result.peak_coords, result.scales,
                                           central_facet_psfs, weights_freq, ws.scale_sigmas, fft_padding,
                                           mask_per_scale);

      is_independant_mask_initialized = true;
    }

    scale::convolve_with_scales(exec_resources, stream_a, scale_ctx, mean_residual, scale_kernels, scales_x_dirty);

    if (activate_dependant_masking)
      common::mask_and_abs_async(stream_a, scales_x_dirty, mask_per_scale, -std::numeric_limits<float>::infinity(),
                                 p.clean_negative);
    else
      common::mask_and_abs_async(stream_a, scales_x_dirty, ws.mask, -std::numeric_limits<float>::infinity(),
                                 p.clean_negative);

    int selected_scale_idx = scale::scale_selection(exec_resources, stream_a, scales_x_dirty, ws.scale_bias,
                                                    scale_stall_tracker.get_all_stalled());

    FD_LOG_INFO("run_wscms: selected scale {}, dependant_masking {}", selected_scale_idx, activate_dependant_masking);

    mean_residual = core::slice_leading(scales_x_dirty, selected_scale_idx);

    core::device_vect<float> sigma_view(ws.scale_sigmas.data_handle() + selected_scale_idx, 1);

    scale::convolve_psfs_with_scale(exec_resources, stream_a, psf_ctx, ws.raw_psfs, sigma_view, selected_scale_idx,
                                    weights_freq, conv_psfs, conv2_psfs);

    std::vector<float> gains;
    if (selected_scale_idx == 0)
      gains = std::vector<float>(n_facets, p.gamma);
    else
      gains = common::compute_gain_batched(exec_resources, stream_a, conv_psfs, weights_freq, p.gamma);

    FD_LOG_DEBUG("run_wscms: gains (n_facets={}): [{:.6f}]", n_facets, fmt::join(gains, ", "));

    float* coeffs_per_chan_ptr = exec_resources.alloc_async<float>(n_freq, stream_a);
    auto coeffs_per_chan = core::device_vect<float>(coeffs_per_chan_ptr, n_freq);

    auto [peak_value, peak_index] = matrix::argmax(peak_ws, mean_residual.data_handle());

    const float threshold = peak_value * p.peak_factor;

    common::mask_less_than_threshold(stream_a, mean_residual, threshold, -std::numeric_limits<float>::infinity());

    FD_LOG_DEBUG("run_wscms: clean loop start scale={} peak={:.8f} threshold={:.8f} max_clean_iter={}",
                 selected_scale_idx, peak_value, threshold, p.max_clean_iteration);

    // Clean loop over mean residual
    int n_clean_iter = 0;
    while (peak_value > threshold && n_clean_iter < p.max_clean_iteration) {
      const auto peak_coords = util::unravel_index_2D(peak_index, dirty_ncols);
      const int facet_idx = ws.map_pixel_facet(peak_coords.first, peak_coords.second);
      const float gain = gains.at(facet_idx);

      result.add_component(peak_coords, selected_scale_idx, gain);

      FD_LOG_DEBUG("run_wscms:   [sub={}] peak={:.8f} at ({},{}) facet={} gain={:.6f}", n_clean_iter, peak_value,
                   peak_coords.first, peak_coords.second, facet_idx, gain);

      core::device_span3d<float> conv_psf = core::slice_leading(conv_psfs, facet_idx);
      core::device_span2d<float> conv2_psf = core::slice_leading(conv2_psfs, facet_idx);

      const std::size_t coeffs_offset = (total_iterations + n_clean_iter) * n_order;
      auto spectral_coeffs = core::device_vect<float>(d_all_coeffs + coeffs_offset, n_order);
      multi_frequency::fit_coefficients(exec_resources, stream_b, dirty, jones_norm, weights_freq, ws.xdes, peak_coords,
                                        spectral_coeffs, coeffs_per_chan);

      common::subtract_component_async(stream_b, dirty, conv_psf, coeffs_per_chan, peak_coords, gain);
      common::subtract_component_async(stream_a, mean_residual, conv2_psf, peak_coords, peak_value * gain);
      std::tie(peak_value, peak_index) = matrix::argmax(peak_ws, mean_residual.data_handle());

      n_clean_iter++;
    }

    stream_a.sync();
    stream_b.sync();

    exec_resources.free_async(coeffs_per_chan_ptr, stream_a);

    FD_LOG_INFO("run_wscms: scale {} produced {} clean iterations", selected_scale_idx, n_clean_iter);

    if (n_clean_iter == 0) {
      FD_LOG_INFO("wscms_minor_cycles: no components found, stopping");
      break;
    }

    total_iterations += n_clean_iter;

    // Reset mean_residual view to the original mean_residual_ptr to store the new mean
    mean_residual = core::device_span2d<float>(mean_residual_ptr, dirty_nrows, dirty_ncols);
    linalg::weighted_sum_async(stream_a, dirty, weights_freq, mean_residual);

    float this_flux = matrix::max(exec_resources, stream_a, mean_residual, ws.mask, p.clean_negative);
    float this_rms = matrix::rms(exec_resources, stream_a, mean_residual, ws.mask);

    FD_LOG_INFO("run_wscms: [iter={}] peak_flux={:.8f} rms={:.8f}", total_iterations, this_flux, this_rms);

    deconv_convergence.track_flux(this_flux, n_clean_iter);
    scale_stall_tracker.update(selected_scale_idx, this_rms);

    FD_LOG_DEBUG("run_wscms: outer iter end delta_flux={:+.8f} delta_rms={:+.8f}", this_flux - track_flux,
                 this_rms - track_rms);

    track_flux = this_flux;
    track_rms = this_rms;

    if (scale_stall_tracker.is_stall(selected_scale_idx))
      FD_LOG_INFO("wscms_minor_cycles: retired scale {} due to stall", selected_scale_idx);
  }

  stream_a.sync();
  stream_b.sync();

  // TODO: deconv_convergence.status => log string
  FD_LOG_INFO("run_wscms: completed ({} iterations, exit={})", deconv_convergence.iteration(),
              static_cast<int>(deconv_convergence.status()));

  result.add_coeffs_from_device(core::device_span2d<float>{d_all_coeffs, total_iterations, n_order});
  result.final_flux = track_flux;
  result.total_iterations = total_iterations;

  exec_resources.free_async(conv2_psfs_ptr, stream_a);
  exec_resources.free_async(conv_psfs_ptr, stream_a);
  exec_resources.free_async(scales_x_dirty_ptr, stream_a);
  exec_resources.free_async(d_all_coeffs, stream_a);
  exec_resources.free_async(scale_kernels_ptr, stream_a);
  exec_resources.free_async(mean_residual_ptr, stream_a);
  if (mask_per_scale_ptr != nullptr) exec_resources.free_async(mask_per_scale_ptr, stream_a);

  return result;
}

}  // namespace fast_deconv::algorithm::wscms