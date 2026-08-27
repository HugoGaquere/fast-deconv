#include <algorithm>
#include <cstdint>
#include <fast_deconv/common/clean.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace {
struct overlap_region {
  int ax0, ay0, bx0, by0;  // Top-left in A and B
  int w, h;                // Overlap size
  int lda, ldb;            // A and B row stride in elements
};

overlap_region compute_overlap_region(int y_center, int x_center, int a_height, int a_width, int b_height, int b_width)
{
  const int bx0 = x_center - b_width / 2;
  const int by0 = y_center - b_height / 2;
  const int ax0 = std::max(bx0, 0);
  const int ay0 = std::max(by0, 0);
  const int ax1 = std::min(bx0 + b_width, a_width);    // exclusive
  const int ay1 = std::min(by0 + b_height, a_height);  // exclusive
  return {ax0, ay0, ax0 - bx0, ay0 - by0, ax1 - ax0, ay1 - ay0, a_width, b_width};
}
}  // namespace

namespace fast_deconv::kernel {

// Subtract PSF from residual across all frequencies at peak location
// residual[f, y, x] -= per_chan[f] * gain * psf[f, y, x]  (within overlap region)
__global__ void subtract_component_kernel(float* residual, const float* psf, const float* per_chan, overlap_region ovr,
                                          float gain, int dirty_freq_stride, int psf_freq_stride)
{
  const int tix = blockIdx.x * blockDim.x + threadIdx.x;
  const int tiy = blockIdx.y * blockDim.y + threadIdx.y;
  const int f = blockIdx.z;
  if (tix >= ovr.w || tiy >= ovr.h) return;

  const std::int64_t r_offset =
      static_cast<std::int64_t>(f) * dirty_freq_stride + (ovr.ay0 + tiy) * ovr.lda + (ovr.ax0 + tix);
  const std::int64_t p_offset =
      static_cast<std::int64_t>(f) * psf_freq_stride + (ovr.by0 + tiy) * ovr.ldb + (ovr.bx0 + tix);

  residual[r_offset] -= per_chan[f] * gain * psf[p_offset];
}

__global__ void subtract_component_kernel(float* residual, const float* psf, overlap_region ovr, float gain)
{
  const int tix = blockIdx.x * blockDim.x + threadIdx.x;
  const int tiy = blockIdx.y * blockDim.y + threadIdx.y;
  if (tix >= ovr.w || tiy >= ovr.h) return;

  const int r_offset = (ovr.ay0 + tiy) * ovr.lda + (ovr.ax0 + tix);
  const int p_offset = (ovr.by0 + tiy) * ovr.ldb + (ovr.bx0 + tix);
  residual[r_offset] -= gain * psf[p_offset];
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::common {

void subtract_component_async(const core::exec_ctx& ctx, core::span2d<float> residual, core::span2d<float> psf,
                              std::pair<int, int> peak_coords, float gain)
{
  const int residual_nrows = residual.extent(0), residual_ncols = residual.extent(1);
  const auto ovr = compute_overlap_region(peak_coords.first, peak_coords.second, residual_nrows, residual_ncols,
                                          psf.extent(0), psf.extent(1));

  dim3 block(32, 8);
  dim3 grid(CEIL_DIV(ovr.w, block.x), CEIL_DIV(ovr.h, block.y));
  kernel::subtract_component_kernel<<<grid, block, 0, ctx.cuda_stream>>>(residual.data_handle(), psf.data_handle(), ovr,
                                                                         gain);
}

void subtract_component_async(const core::exec_ctx& ctx, core::span3d<float> residual, core::span3d<float> psf,
                              core::span1d<float> spectral_coeffs, std::pair<int, int> peak_coords, float gain)
{
  const int n_freq = residual.extent(0);
  const int residual_nrows = residual.extent(1);
  const int residual_ncols = residual.extent(2);
  const int residual_freq_stride = residual_nrows * residual_ncols;
  const int psf_nrows = psf.extent(1);
  const int psf_ncols = psf.extent(2);
  const int psf_freq_stride = psf_nrows * psf_ncols;

  const auto ovr = compute_overlap_region(peak_coords.first, peak_coords.second, residual_nrows, residual_ncols,
                                          psf_nrows, psf_ncols);

  dim3 block(32, 8, 1);
  dim3 grid(CEIL_DIV(ovr.w, block.x), CEIL_DIV(ovr.h, block.y), n_freq);
  kernel::subtract_component_kernel<<<grid, block, 0, ctx.cuda_stream>>>(residual.data_handle(), psf.data_handle(),
                                                                         spectral_coeffs.data_handle(), ovr, gain,
                                                                         residual_freq_stride, psf_freq_stride);
}

}  // namespace fast_deconv::common
