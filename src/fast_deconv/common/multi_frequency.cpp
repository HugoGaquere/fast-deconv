#include <cassert>
#include <cstdint>
#include <fast_deconv/common/detail/spectral_fit.hpp>
#include <fast_deconv/common/multi_frequency.hpp>
#include <fast_deconv/core/macro.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fd_backend/core/for_each.hpp>
#include <stdexcept>

namespace fast_deconv::multi_frequency {

void fit_coefficients(const core::exec_ctx& ctx, core::span3d<const float> residual,
                      core::span3d<const float> jones_norm, core::span1d<const float> weights_freq,
                      core::span2d<const float> xdes, core::index2d peak_coords,
                      core::span1d<float> compact_coeffs_out, core::span1d<float> coeffs_per_chan_out)
{
  FD_PROFILE_FN();
  assert(xdes.is_exhaustive());
  const std::int64_t n_freq = xdes.extent(0);
  const std::int64_t n_order = xdes.extent(1);
  const auto [peak_row, peak_col] = peak_coords;

  if (n_order > detail::kSpectralMaxOrder)
    throw std::invalid_argument("fit_coefficients: n_order exceeds SPECTRAL_MAX_ORDER");

  const std::int64_t residual_freq_stride = residual.stride(0);
  const std::int64_t residual_peak_offset = residual.mapping()(0, peak_row, peak_col);
  const std::int64_t jn_freq_stride = jones_norm.stride(0);
  const std::int64_t jn_peak_offset = jones_norm.mapping()(0, peak_row, peak_col);
  float* compact = compact_coeffs_out.data_handle();
  float* per_chan = coeffs_per_chan_out.data_handle();
  const float* res = residual.data_handle();
  const float* x = xdes.data_handle();
  const float* jn = jones_norm.data_handle();
  const float* w = weights_freq.data_handle();

  // One index: the solve is tiny and sequential, so on CUDA the launch dominates. A singular system leaves NaN in
  // both outputs.
  core::for_each_index(ctx, 1, [=] FD_HOST_DEVICE(std::int64_t) {
    detail::spectral_fit(compact, per_chan, res, x, jn, w, n_freq, n_order, residual_freq_stride, residual_peak_offset,
                         jn_freq_stride, jn_peak_offset);
  });
}

}  // namespace fast_deconv::multi_frequency
