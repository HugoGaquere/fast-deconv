#include <cstdint>
#include <fast_deconv/common/clean.hpp>
#include <fast_deconv/common/region.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::kernel {

// Subtract PSF from residual across all frequencies at peak location
// residual[f, row, col] -= per_chan[f] * gain * psf[f, row, col]  (within overlap region)
__global__ void subtract_component_kernel(float* residual, const float* psf, const float* per_chan,
                                          common::overlap_region ovr, float gain, int dirty_freq_stride,
                                          int psf_freq_stride)
{
  const int tix = blockIdx.x * blockDim.x + threadIdx.x;
  const int tiy = blockIdx.y * blockDim.y + threadIdx.y;
  const int f = blockIdx.z;
  if (tix >= ovr.ncol || tiy >= ovr.nrow) return;

  const std::int64_t r_offset =
      static_cast<std::int64_t>(f) * dirty_freq_stride + (ovr.arow0 + tiy) * ovr.lda + (ovr.acol0 + tix);
  const std::int64_t p_offset =
      static_cast<std::int64_t>(f) * psf_freq_stride + (ovr.brow0 + tiy) * ovr.ldb + (ovr.bcol0 + tix);

  residual[r_offset] -= per_chan[f] * gain * psf[p_offset];
}

__global__ void subtract_component_kernel(float* residual, const float* psf, common::overlap_region ovr, float gain)
{
  const int tix = blockIdx.x * blockDim.x + threadIdx.x;
  const int tiy = blockIdx.y * blockDim.y + threadIdx.y;
  if (tix >= ovr.ncol || tiy >= ovr.nrow) return;

  const int r_offset = (ovr.arow0 + tiy) * ovr.lda + (ovr.acol0 + tix);
  const int p_offset = (ovr.brow0 + tiy) * ovr.ldb + (ovr.bcol0 + tix);
  residual[r_offset] -= gain * psf[p_offset];
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::common {

void subtract_component_async(const core::exec_ctx& ctx, core::span2d<float> residual, core::span2d<float> psf,
                              index2d peak_coords, float gain)
{
  const int residual_nrows = residual.extent(0), residual_ncols = residual.extent(1);
  const auto ovr = compute_overlap_region(peak_coords, residual_nrows, residual_ncols, psf.extent(0), psf.extent(1));

  dim3 block(32, 8);
  dim3 grid(CEIL_DIV(ovr.ncol, block.x), CEIL_DIV(ovr.nrow, block.y));
  kernel::subtract_component_kernel<<<grid, block, 0, ctx.cuda_stream>>>(residual.data_handle(), psf.data_handle(), ovr,
                                                                         gain);
}

void subtract_component_async(const core::exec_ctx& ctx, core::span3d<float> residual, core::span3d<float> psf,
                              core::span1d<float> spectral_coeffs, index2d peak_coords, float gain)
{
  const int n_freq = residual.extent(0);
  const int residual_nrows = residual.extent(1);
  const int residual_ncols = residual.extent(2);
  const int residual_freq_stride = residual_nrows * residual_ncols;
  const int psf_nrows = psf.extent(1);
  const int psf_ncols = psf.extent(2);
  const int psf_freq_stride = psf_nrows * psf_ncols;

  const auto ovr = compute_overlap_region(peak_coords, residual_nrows, residual_ncols, psf_nrows, psf_ncols);

  dim3 block(32, 8, 1);
  dim3 grid(CEIL_DIV(ovr.ncol, block.x), CEIL_DIV(ovr.nrow, block.y), n_freq);
  kernel::subtract_component_kernel<<<grid, block, 0, ctx.cuda_stream>>>(residual.data_handle(), psf.data_handle(),
                                                                         spectral_coeffs.data_handle(), ovr, gain,
                                                                         residual_freq_stride, psf_freq_stride);
}

}  // namespace fast_deconv::common
