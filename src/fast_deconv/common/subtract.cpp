#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fast_deconv/common/subtract.hpp>
#include <fast_deconv/core/macro.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fd_backend/core/for_each.hpp>

namespace fast_deconv::common {

namespace {

// Intersection of a small array B centered on an index of a large array A.
struct overlap_region {
  std::int64_t arow0, acol0, brow0, bcol0;  // Top-left in A and B
  std::int64_t nrow, ncol;                  // Overlap size
  std::int64_t lda, ldb;                    // A and B row stride in elements
};

// Clamps B's footprint, centered at center in A, to A's bounds.
overlap_region compute_overlap_region(core::index2d center, std::int64_t a_nrow, std::int64_t a_ncol,
                                      std::int64_t b_nrow, std::int64_t b_ncol)
{
  const std::int64_t brow_origin = center.row - b_nrow / 2;
  const std::int64_t bcol_origin = center.col - b_ncol / 2;
  const std::int64_t arow0 = std::max<std::int64_t>(brow_origin, 0);
  const std::int64_t acol0 = std::max<std::int64_t>(bcol_origin, 0);
  const std::int64_t arow1 = std::min(brow_origin + b_nrow, a_nrow);  // exclusive
  const std::int64_t acol1 = std::min(bcol_origin + b_ncol, a_ncol);  // exclusive
  return {arow0, acol0, arow0 - brow_origin, acol0 - bcol_origin, arow1 - arow0, acol1 - acol0, a_ncol, b_ncol};
}

}  // namespace

void subtract_component(const core::exec_ctx& ctx, core::span2d<float> residual, core::span2d<const float> psf,
                        core::index2d peak_coords, float gain, matrix::peak_criterion criterion, float threshold)
{
  FD_PROFILE_FN();
  assert(residual.is_exhaustive() && psf.is_exhaustive());
  const auto ovr =
      compute_overlap_region(peak_coords, residual.extent(0), residual.extent(1), psf.extent(0), psf.extent(1));
  float* res = residual.data_handle();
  const float* p = psf.data_handle();
  core::for_each_index(ctx, ovr.nrow, ovr.ncol, [=] FD_HOST_DEVICE(std::int64_t r, std::int64_t c) {
    const std::int64_t i = (ovr.arow0 + r) * ovr.lda + ovr.acol0 + c;
    const float x = res[i];
    if (criterion(x, i) >= threshold) res[i] = x - gain * p[(ovr.brow0 + r) * ovr.ldb + ovr.bcol0 + c];
  });
}

void subtract_component(const core::exec_ctx& ctx, core::span3d<float> residual, core::span3d<const float> psf,
                        core::span1d<const float> spectral_coeffs, core::index2d peak_coords, float gain)
{
  FD_PROFILE_FN();
  const auto ovr =
      compute_overlap_region(peak_coords, residual.extent(1), residual.extent(2), psf.extent(1), psf.extent(2));
  const std::int64_t res_freq_stride = residual.stride(0);
  const std::int64_t psf_freq_stride = psf.stride(0);
  float* res = residual.data_handle();
  const float* p = psf.data_handle();
  const float* coeffs = spectral_coeffs.data_handle();
  core::for_each_index(ctx, residual.extent(0), ovr.nrow, ovr.ncol,
                       [=] FD_HOST_DEVICE(std::int64_t f, std::int64_t r, std::int64_t c) {
                         res[f * res_freq_stride + (ovr.arow0 + r) * ovr.lda + ovr.acol0 + c] -=
                             gain * coeffs[f] * p[f * psf_freq_stride + (ovr.brow0 + r) * ovr.ldb + ovr.bcol0 + c];
                       });
}

}  // namespace fast_deconv::common
