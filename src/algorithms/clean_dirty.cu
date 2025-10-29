#include "clean_dirty.cuh"

namespace fast_deconv::wscms {

void _clean_dirty_kernel(float *dirty, float *psf, uint dirty_width,
                         uint psf_width, uint patch_width, Rect dirty_patch,
                         Rect psf_patch, float gain, float coeffs) {

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N;
       idx += blockDim.x * gridDim.x) {
    uint row = idx / patch_width;
    uint col = idx % patch_width;
    uint dirty_idx =
        (dirty_patch.y0 + row) * dirty_width + (dirty_patch.x0 + col);
    uint psf_idx = (psf_patch.y0 + row) * psf_width + (psf_patch.x0 + col);
    dirty[dirty_idx] -= psf[psf_idx] * coeffs * gain;
  }
}

void clean_dirty_async(float *dirty, float *psf, uint dirty_width,
                       uint psf_width, Rect dirty_patch, Rect psf_patch,
                       float gain, float coeffs, cudaStream_t stream) {
  // FIXME: Fine tune conf
  _clean_dirty_kernel<<<32 * 40, 256, 0, stream>>>(
      dirty, psf, dirty_width, psf_width, dirty_patch.width(), dirty_patch,
      psf_patch, gain, coeffs);
}

} // namespace fast_deconv::wscms
