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

// Row-wise vectorized kernel using float4 loads/stores
// One block per row - handles strided data where stride[2] is not aligned to 4
// but stride[3] == 1 (innermost dimension is contiguous within each row)
__global__ void subtract_psf_from_dirty_kernel_row_vec4(const float* __restrict__ psf_ptr,
                                                        const float* __restrict__ dirty_ptr,
                                                        const float* __restrict__ coeffs,
                                                        float* __restrict__ out_ptr,
                                                        float gain,
                                                        size_t n_channels,
                                                        size_t n_pol,
                                                        size_t height,
                                                        size_t width,
                                                        size_t stride_ch,
                                                        size_t stride_pol,
                                                        size_t stride_row)
{
  // Each block handles one row
  const size_t row_idx    = blockIdx.x;
  const size_t total_rows = n_channels * n_pol * height;
  if (row_idx >= total_rows) return;

  // Decode row index to (ch, pol, h)
  const size_t ch  = row_idx / (n_pol * height);
  const size_t rem = row_idx % (n_pol * height);
  const size_t pol = rem / height;
  const size_t h   = rem % height;

  // Base offset for this row (computed from actual strides)
  const size_t row_base = ch * stride_ch + pol * stride_pol + h * stride_row;
  const float coeff     = coeffs[ch] * gain;

  // Vectorized portion: process 4 elements per iteration
  const size_t vec_width = width / 4;
  for (size_t vec_i = threadIdx.x; vec_i < vec_width; vec_i += blockDim.x) {
    const size_t offset = row_base + vec_i * 4;

    const float4 psf_v   = *reinterpret_cast<const float4*>(psf_ptr + offset);
    const float4 dirty_v = *reinterpret_cast<const float4*>(dirty_ptr + offset);

    float4 out_v;
    out_v.x = dirty_v.x - psf_v.x * coeff;
    out_v.y = dirty_v.y - psf_v.y * coeff;
    out_v.z = dirty_v.z - psf_v.z * coeff;
    out_v.w = dirty_v.w - psf_v.w * coeff;

    *reinterpret_cast<float4*>(out_ptr + offset) = out_v;
  }

  // Remainder: handle width % 4 elements
  const size_t remainder_start = vec_width * 4;
  for (size_t i = remainder_start + threadIdx.x; i < width; i += blockDim.x) {
    const size_t offset = row_base + i;
    out_ptr[offset]     = dirty_ptr[offset] - psf_ptr[offset] * coeff;
  }
}

// Row-wise vectorized kernel using float2 loads/stores
// For when stride_row % 2 == 0 but stride_row % 4 != 0
__global__ void subtract_psf_from_dirty_kernel_row_vec2(const float* __restrict__ psf_ptr,
                                                        const float* __restrict__ dirty_ptr,
                                                        const float* __restrict__ coeffs,
                                                        float* __restrict__ out_ptr,
                                                        float gain,
                                                        size_t n_channels,
                                                        size_t n_pol,
                                                        size_t height,
                                                        size_t width,
                                                        size_t stride_ch,
                                                        size_t stride_pol,
                                                        size_t stride_row)
{
  const size_t row_idx    = blockIdx.x;
  const size_t total_rows = n_channels * n_pol * height;
  if (row_idx >= total_rows) return;

  const size_t ch  = row_idx / (n_pol * height);
  const size_t rem = row_idx % (n_pol * height);
  const size_t pol = rem / height;
  const size_t h   = rem % height;

  const size_t row_base = ch * stride_ch + pol * stride_pol + h * stride_row;
  const float coeff     = coeffs[ch] * gain;

  // Vectorized portion: process 2 elements per iteration
  const size_t vec_width = width / 2;
  for (size_t vec_i = threadIdx.x; vec_i < vec_width; vec_i += blockDim.x) {
    const size_t offset = row_base + vec_i * 2;

    const float2 psf_v   = *reinterpret_cast<const float2*>(psf_ptr + offset);
    const float2 dirty_v = *reinterpret_cast<const float2*>(dirty_ptr + offset);

    float2 out_v;
    out_v.x = dirty_v.x - psf_v.x * coeff;
    out_v.y = dirty_v.y - psf_v.y * coeff;

    *reinterpret_cast<float2*>(out_ptr + offset) = out_v;
  }

  // Remainder: handle width % 2 elements (at most 1)
  if (width % 2 != 0 && threadIdx.x == 0) {
    const size_t offset = row_base + width - 1;
    out_ptr[offset]     = dirty_ptr[offset] - psf_ptr[offset] * coeff;
  }
}

// Coalesced scalar kernel - one block per row, threads access consecutive elements
// For when stride[3] == 1 but stride_row is odd (can't use float2/float4)
__global__ void subtract_psf_from_dirty_kernel_row_coalesced(const float* __restrict__ psf_ptr,
                                                             const float* __restrict__ dirty_ptr,
                                                             const float* __restrict__ coeffs,
                                                             float* __restrict__ out_ptr,
                                                             float gain,
                                                             size_t n_channels,
                                                             size_t n_pol,
                                                             size_t height,
                                                             size_t width,
                                                             size_t stride_ch,
                                                             size_t stride_pol,
                                                             size_t stride_row)
{
  const size_t row_idx    = blockIdx.x;
  const size_t total_rows = n_channels * n_pol * height;
  if (row_idx >= total_rows) return;

  const size_t ch  = row_idx / (n_pol * height);
  const size_t rem = row_idx % (n_pol * height);
  const size_t pol = rem / height;
  const size_t h   = rem % height;

  const size_t row_base = ch * stride_ch + pol * stride_pol + h * stride_row;
  const float coeff     = coeffs[ch] * gain;

  // Each thread processes elements with stride of blockDim.x
  // Consecutive threads access consecutive memory addresses (coalesced)
  for (size_t i = threadIdx.x; i < width; i += blockDim.x) {
    const size_t offset = row_base + i;
    out_ptr[offset]     = dirty_ptr[offset] - psf_ptr[offset] * coeff;
  }
}

void subtract_psf_from_dirty_async(core::device_span4d_fs& psf,
                                   core::device_span4d_fs& dirty,
                                   core::device_vect_f& coeffs,
                                   core::device_span4d_fs& out,
                                   float gain,
                                   core::stream_resources& resources)
{
  // Check if innermost dimension is contiguous (stride[3] == 1) for all spans
  const bool innermost_contiguous = psf.mapping().stride(3) == 1 &&
                                    dirty.mapping().stride(3) == 1 &&
                                    out.mapping().stride(3) == 1;

  // fmt::println("Innermost_contiguous {}", innermost_contiguous);


  if (!innermost_contiguous || true) {
    // Fall back to naive kernel for non-contiguous innermost dimension
    subtract_psf_from_dirty_kernel_naive<<<CEIL_DIV(out.size(), 256), 256, 0, resources.stream>>>(
      psf, dirty, coeffs, out, gain);
    CHECK_LAST_CUDA_ERROR();
    return;
  }

  // Innermost is contiguous - use row-wise kernel with appropriate vectorization
  const size_t n_channels = out.extent(0);
  const size_t n_pol      = out.extent(1);
  const size_t height     = out.extent(2);
  const size_t width      = out.extent(3);
  const size_t stride_ch  = out.mapping().stride(0);
  const size_t stride_pol = out.mapping().stride(1);
  const size_t stride_row = out.mapping().stride(2);

  const size_t total_rows  = n_channels * n_pol * height;
  constexpr int block_size = 256;

  if (stride_row % 4 == 0 && width >= 4) {
    fmt::println("float4 vect");
    // Best case: float4 vectorization (128-bit loads/stores)
    subtract_psf_from_dirty_kernel_row_vec4<<<total_rows, block_size, 0, resources.stream>>>(
      psf.data_handle(), dirty.data_handle(), coeffs.data_handle(), out.data_handle(), gain,
      n_channels, n_pol, height, width, stride_ch, stride_pol, stride_row);
  } else if (stride_row % 2 == 0 && width >= 2) {
    fmt::println("float2 vect");
    // Good case: float2 vectorization (64-bit loads/stores)
    subtract_psf_from_dirty_kernel_row_vec2<<<total_rows, block_size, 0, resources.stream>>>(
      psf.data_handle(), dirty.data_handle(), coeffs.data_handle(), out.data_handle(), gain,
      n_channels, n_pol, height, width, stride_ch, stride_pol, stride_row);
  } else {
    fmt::println("fallback coalesced");
    // Fallback: coalesced scalar (still better than naive due to coalescing)
    subtract_psf_from_dirty_kernel_row_coalesced<<<total_rows, block_size, 0, resources.stream>>>(
      psf.data_handle(), dirty.data_handle(), coeffs.data_handle(), out.data_handle(), gain,
      n_channels, n_pol, height, width, stride_ch, stride_pol, stride_row);
  }
  CHECK_LAST_CUDA_ERROR();
}

}  // namespace fast_deconv::algo::wscms::detail
