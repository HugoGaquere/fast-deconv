#pragma once

#include <thrust/tuple.h>

#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::algo::wscms::detail {

template <class Mdspan>
__device__ inline auto linear_to_4d_clean(typename Mdspan::size_type lin, const Mdspan& m)
{
  using index_t = typename Mdspan::index_type;

  index_t i3 = lin % m.extent(3);
  lin /= m.extent(3);
  index_t i2 = lin % m.extent(2);
  lin /= m.extent(2);
  index_t i1 = lin % m.extent(1);
  lin /= m.extent(1);
  index_t i0 = lin;

  return thrust::make_tuple(i0, i1, i2, i3);
}

// Fused kernel that performs:
// 1. dirty[i] -= psf[i] * coeffs[ch] * gain
// 2. scaled_dirty[i] -= psf_2[i] * gain_scaled * mask[i]
__global__ void clean_dirties_kernel(core::device_span4d_fs psf,
                                      core::device_span4d_fs psf_2,
                                      core::device_span4d_fs dirty,
                                      core::device_span4d_fs scaled_dirty,
                                      core::device_vect_f coeffs,
                                      core::device_span4d_fs mask,
                                      float gain)
{
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= dirty.size()) return;

  auto [i0, i1, i2, i3] = linear_to_4d_clean(tid, dirty);

  // Subtract PSF from dirty: dirty -= psf * coeffs[ch] * gain
  dirty(i0, i1, i2, i3) -= psf(i0, i1, i2, i3) * coeffs(i0) * gain;

  // Subtract PSF from scaled_dirty with mask: scaled_dirty -= psf_2 * gain_scaled * mask
  const float mask_val = mask(i0, i1, i2, i3);
  scaled_dirty(i0, i1, i2, i3) -= psf_2(i0, i1, i2, i3) * gain * mask_val;
}

void clean_dirties_async(core::device_span4d_fs& psf,
                          core::device_span4d_fs& psf_2,
                          core::device_span4d_fs& dirty,
                          core::device_span4d_fs& scaled_dirty,
                          core::device_vect_f& coeffs,
                          core::device_span4d_fs& mask,
                          float gain,
                          core::stream_resources& resources)
{
  clean_dirties_kernel<<<CEIL_DIV(dirty.size(), 256), 256, 0, resources.stream>>>(
    psf, psf_2, dirty, scaled_dirty, coeffs, mask, gain);
  CHECK_LAST_CUDA_ERROR();
}

}  // namespace fast_deconv::algo::wscms::detail
