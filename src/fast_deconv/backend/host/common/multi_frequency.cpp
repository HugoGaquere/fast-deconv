#include <cassert>
#include <cstdint>
#include <fast_deconv/common/detail/spectral_fit.hpp>
#include <fast_deconv/common/multi_frequency.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <stdexcept>

namespace fast_deconv::multi_frequency {

void fit_coefficients(const core::exec_ctx& /*ctx*/, core::span3d<const float> residual,
                      core::span3d<const float> jones_norm, core::span1d<const float> weights_freq,
                      core::span2d<const float> xdes, common::index2d peak_coords,
                      core::span1d<float> compact_coeffs_out, core::span1d<float> coeffs_per_chan_out)
{
  FD_PROFILE_FN();
  assert(xdes.is_exhaustive());
  const std::int64_t n_freq = xdes.extent(0);
  const std::int64_t n_order = xdes.extent(1);
  const auto [peak_row, peak_col] = peak_coords;

  if (n_order > detail::kSpectralMaxOrder)
    throw std::invalid_argument("fit_coefficients: n_order exceeds SPECTRAL_MAX_ORDER");

  const bool solved =
      detail::spectral_fit(compact_coeffs_out.data_handle(), coeffs_per_chan_out.data_handle(), residual.data_handle(),
                           xdes.data_handle(), jones_norm.data_handle(), weights_freq.data_handle(), n_freq, n_order,
                           residual.mapping().stride(0), residual.mapping()(0, peak_row, peak_col),
                           jones_norm.mapping().stride(0), jones_norm.mapping()(0, peak_row, peak_col));
  if (!solved) throw std::domain_error("fit_coefficients: singular spectral system");
}

}  // namespace fast_deconv::multi_frequency
