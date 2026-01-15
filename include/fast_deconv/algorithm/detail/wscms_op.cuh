#pragma once

#include <thrust/tuple.h>

#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace {
template <class Mdspan>
__device__ inline auto linear_to_4d(typename Mdspan::size_type lin, const Mdspan& m)
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

}  // namespace

namespace fast_deconv::algo::wscms::detail {

__global__ void subtract_psf_from_dirty_kernel_naive(core::device_span4d_fs psf,
                                                     core::device_span4d_fs dirty,
                                                     core::device_vect_f coeffs,
                                                     core::device_span4d_fs out,
                                                     float gain)
{
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= out.size()) return;
  auto [i0, i1, i2, i3] = linear_to_4d(tid, out);

  out(i0, i1, i2, i3) = dirty(i0, i1, i2, i3) - psf(i0, i1, i2, i3) * coeffs(i0) * gain;
}

void subtract_psf_from_dirty_async(core::device_span4d_fs& psf,
                             core::device_span4d_fs& dirty,
                             core::device_vect_f& coeffs,
                             core::device_span4d_fs& out,
                             float gain,
                             core::stream_resources& resources)
{
  subtract_psf_from_dirty_kernel_naive<<<CEIL_DIV(out.size(), 256), 256, 0, resources.stream>>>(
    psf, dirty, coeffs, out, gain);
  CHECK_LAST_CUDA_ERROR();
}

}  // namespace fast_deconv::algo::wscms::detail
