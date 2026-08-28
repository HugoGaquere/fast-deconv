#include <fast_deconv/common/clean.hpp>

namespace fast_deconv::common {

void subtract_component_async(const core::exec_ctx& ctx, core::span2d<float> residual, core::span2d<float> psf,
                              index2d peak_coords, float gain)
{
  auto ovr = compute_overlap_region(peak_coords, residual.extent(0), residual.extent(1), psf.extent(0), psf.extent(1));

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

  for (int f = 0; f < n_freq; f++) {
    float coeff = gain * spectral_coeffs(f);
    for (int r = 0; r < ovr.nrow; r++) {
      for (int c = 0; c < ovr.ncol; c++) {
        residual(f, ovr.arow0 + r, ovr.acol0 + c) -= coeff * psf(f, ovr.brow0 + r, ovr.bcol0 + c);
      }
    }
  }
}

}  // namespace fast_deconv::common
