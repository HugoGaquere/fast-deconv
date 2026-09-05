#include <fast_deconv/common/clean.hpp>

namespace fast_deconv::common {

namespace {
// Below this the fork costs more than the subtract: a footprint can be a few rows.
constexpr long kMinParallelPixels = 64 * 1024;
}  // namespace

void subtract_component_async(const core::exec_ctx& ctx, core::span2d<float> residual, core::span2d<float> psf,
                              index2d peak_coords, float gain)
{
  auto ovr = compute_overlap_region(peak_coords, residual.extent(0), residual.extent(1), psf.extent(0), psf.extent(1));

#pragma omp parallel for if (static_cast<long>(ovr.nrow) * ovr.ncol > kMinParallelPixels)
  for (int r = 0; r < ovr.nrow; r++) {
    for (int c = 0; c < ovr.ncol; c++) {
      residual(ovr.arow0 + r, ovr.acol0 + c) -= gain * psf(ovr.brow0 + r, ovr.bcol0 + c);
    }
  }
}

void subtract_component_async(const core::exec_ctx& ctx, core::span3d<float> residual, core::span3d<float> psf,
                              core::span1d<float> spectral_coeffs, index2d peak_coords, float gain)
{
  int n_freq = residual.extent(0);
  auto ovr = compute_overlap_region(peak_coords, residual.extent(1), residual.extent(2), psf.extent(1), psf.extent(2));

  // Collapsed: n_freq alone is a handful of channels, too few to fill the pool.
#pragma omp parallel for collapse(2) if (static_cast<long>(n_freq) * ovr.nrow * ovr.ncol > kMinParallelPixels)
  for (int f = 0; f < n_freq; f++) {
    for (int r = 0; r < ovr.nrow; r++) {
      const float coeff = gain * spectral_coeffs(f);
      for (int c = 0; c < ovr.ncol; c++) {
        residual(f, ovr.arow0 + r, ovr.acol0 + c) -= coeff * psf(f, ovr.brow0 + r, ovr.bcol0 + c);
      }
    }
  }
}

}  // namespace fast_deconv::common
