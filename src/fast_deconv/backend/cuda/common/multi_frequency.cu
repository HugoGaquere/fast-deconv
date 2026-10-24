#include <cstdint>
#include <fast_deconv/common/detail/spectral_fit.hpp>
#include <fast_deconv/common/multi_frequency.hpp>
#include <fd_backend/core/launch.cuh>
#include <stdexcept>

namespace fast_deconv::kernel {

// One block, one thread: this is a tiny sequential solve, and the launch dominates. A singular system leaves NaN
// in both outputs.
__global__ void spectral_fit_kernel(float* compact_out, float* per_chan_out, const float* residual, const float* xdes,
                                    const float* jones_norm, const float* weights, std::int64_t n_freq,
                                    std::int64_t n_order, std::int64_t residual_freq_stride,
                                    std::int64_t residual_peak_offset, std::int64_t jn_freq_stride,
                                    std::int64_t jn_peak_offset)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  multi_frequency::detail::spectral_fit(compact_out, per_chan_out, residual, xdes, jones_norm, weights, n_freq, n_order,
                                        residual_freq_stride, residual_peak_offset, jn_freq_stride, jn_peak_offset);
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::multi_frequency {

void fit_coefficients(const core::exec_ctx& ctx, core::span3d<const float> residual,
                      core::span3d<const float> jones_norm, core::span1d<const float> weights_freq,
                      core::span2d<const float> xdes, common::index2d peak_coords,
                      core::span1d<float> compact_coeffs_out, core::span1d<float> coeffs_per_chan_out)
{
  const std::int64_t n_freq = xdes.extent(0);
  const std::int64_t n_order = xdes.extent(1);
  const auto [peak_row, peak_col] = peak_coords;

  if (n_order > detail::kSpectralMaxOrder)
    throw std::invalid_argument("fit_coefficients: n_order exceeds SPECTRAL_MAX_ORDER");

  const std::int64_t jn_freq_stride = jones_norm.mapping().stride(0);
  const std::int64_t jn_peak_offset = jones_norm.mapping()(0, peak_row, peak_col);
  const std::int64_t dirty_freq_stride = residual.mapping().stride(0);
  const std::int64_t dirty_peak_offset = residual.mapping()(0, peak_row, peak_col);

  const auto cfg = core::launch_cfg::custom(1, 1);
  core::launch(ctx, cfg, kernel::spectral_fit_kernel, compact_coeffs_out.data_handle(),
               coeffs_per_chan_out.data_handle(), residual.data_handle(), xdes.data_handle(), jones_norm.data_handle(),
               weights_freq.data_handle(), n_freq, n_order, dirty_freq_stride, dirty_peak_offset, jn_freq_stride,
               jn_peak_offset);
}

}  // namespace fast_deconv::multi_frequency
