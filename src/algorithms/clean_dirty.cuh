#include "fast_deconv/rect.h"
#include <driver_types.h>

namespace fast_deconv::wscms {

void clean_dirty_async(float *dirty, float *psf, uint dirty_width,
                       uint psf_width, Rect dirty_patch, Rect psf_patch,
                       float gain, float coeffs, cudaStream_t stream);
}
