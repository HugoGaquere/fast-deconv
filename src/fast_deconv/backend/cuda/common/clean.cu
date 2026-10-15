#include <cstdint>
#include <fast_deconv/common/clean.hpp>
#include <fast_deconv/common/region.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::kernel {

// Subtract PSF from residual across all frequencies at peak location
// residual[f, row, col] -= per_chan[f] * gain * psf[f, row, col]  (within overlap region)
__global__ void subtract_component_kernel(float* residual, const float* psf, const float* per_chan,
                                          common::overlap_region ovr, float gain, std::int64_t dirty_freq_stride,
                                          std::int64_t psf_freq_stride)
{
  const std::int64_t tix = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  const std::int64_t tiy = std::int64_t{blockIdx.y} * blockDim.y + threadIdx.y;
  const std::int64_t f = blockIdx.z;
  if (tix >= ovr.ncol || tiy >= ovr.nrow) return;

  const std::int64_t r_offset = f * dirty_freq_stride + (ovr.arow0 + tiy) * ovr.lda + (ovr.acol0 + tix);
  const std::int64_t p_offset = f * psf_freq_stride + (ovr.brow0 + tiy) * ovr.ldb + (ovr.bcol0 + tix);

  residual[r_offset] -= per_chan[f] * gain * psf[p_offset];
}

__global__ void subtract_component_kernel(float* residual, const float* psf, common::overlap_region ovr, float gain,
                                          matrix::peak_criterion criterion, float threshold)
{
  const std::int64_t tix = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  const std::int64_t tiy = std::int64_t{blockIdx.y} * blockDim.y + threadIdx.y;
  if (tix >= ovr.ncol || tiy >= ovr.nrow) return;

  const std::int64_t r_offset = (ovr.arow0 + tiy) * ovr.lda + (ovr.acol0 + tix);
  const std::int64_t p_offset = (ovr.brow0 + tiy) * ovr.ldb + (ovr.bcol0 + tix);
  const float x = residual[r_offset];
  if (criterion(x, r_offset) >= threshold) residual[r_offset] = x - gain * psf[p_offset];
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::common {

void subtract_component_async(const core::exec_ctx& ctx, core::span2d<float> residual, core::span2d<const float> psf,
                              index2d peak_coords, float gain, matrix::peak_criterion criterion, float threshold)
{
  const auto ovr =
      compute_overlap_region(peak_coords, residual.extent(0), residual.extent(1), psf.extent(0), psf.extent(1));

  dim3 block(32, 8);
  dim3 grid(core::narrow<unsigned>(core::ceil_div(ovr.ncol, block.x)),
            core::narrow<unsigned>(core::ceil_div(ovr.nrow, block.y)));
  kernel::subtract_component_kernel<<<grid, block, 0, ctx.cuda_stream>>>(residual.data_handle(), psf.data_handle(), ovr,
                                                                         gain, criterion, threshold);
}

void subtract_component_async(const core::exec_ctx& ctx, core::span3d<float> residual, core::span3d<const float> psf,
                              core::span1d<const float> spectral_coeffs, index2d peak_coords, float gain)
{
  const auto ovr =
      compute_overlap_region(peak_coords, residual.extent(1), residual.extent(2), psf.extent(1), psf.extent(2));

  dim3 block(32, 8, 1);
  dim3 grid(core::narrow<unsigned>(core::ceil_div(ovr.ncol, block.x)),
            core::narrow<unsigned>(core::ceil_div(ovr.nrow, block.y)), core::narrow<unsigned>(residual.extent(0)));
  kernel::subtract_component_kernel<<<grid, block, 0, ctx.cuda_stream>>>(
      residual.data_handle(), psf.data_handle(), spectral_coeffs.data_handle(), ovr, gain, residual.mapping().stride(0),
      psf.mapping().stride(0));
}

}  // namespace fast_deconv::common
