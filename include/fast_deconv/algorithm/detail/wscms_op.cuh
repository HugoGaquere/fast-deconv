#pragma once

#include <thrust/tuple.h>

#include <fast_deconv/core/access_policy.hpp>
#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fast_deconv/util/mdspan_utils.hpp>

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

__global__ void subtract_psf_from_dirty_kernel_naive(core::device_span4d_S<float> psf,
                                                     core::device_span4d_S<float> dirty,
                                                     core::device_vect<float> coeffs,
                                                     core::device_span4d_S<float> out,
                                                     float gain)
{
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= out.size()) return;
  auto [i0, i1, i2, i3] = linear_to_4d(tid, out);

  out(i0, i1, i2, i3) = dirty(i0, i1, i2, i3) - psf(i0, i1, i2, i3) * coeffs(i0) * gain;
}

template <typename VectType, typename ScalarType, int ITEMS_PER_THREAD>
__global__ void subtract_psf_from_dirty_kernel_vect(core::device_span4d_S<float> psf,
                                                    core::device_span4d_S<float> dirty,
                                                    core::device_vect<float> coeffs,
                                                    core::device_span4d_S<float> out,
                                                    float gain,
                                                    uint nb_rows,
                                                    uint nb_cols)
{
  const size_t row = blockIdx.y + gridDim.y * blockIdx.z;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x * ITEMS_PER_THREAD;
  if (row >= nb_rows || col >= nb_cols) return;

  const int rem_in_row = static_cast<int>(nb_cols - col);
  const int lanes      = min(ITEMS_PER_THREAD, rem_in_row);
  const size_t lin     = row * nb_cols + col;

  // idx for psf and dirty
  const auto indices_md = util::linear_to_indices(lin, psf);

  // Compute all pointers upfront
  const auto* psf_ptr =
    psf.data_handle() + std::apply([&psf](auto... i) { return psf.mapping()(i...); }, indices_md);
  const auto* dirty_ptr =
    dirty.data_handle() +
    std::apply([&dirty](auto... i) { return dirty.mapping()(i...); }, indices_md);
  auto* out_ptr =
    out.data_handle() + std::apply([&out](auto... i) { return out.mapping()(i...); }, indices_md);

  // Load vect
  const auto psf_vect   = *reinterpret_cast<const VectType*>(psf_ptr);
  const auto dirty_vect = *reinterpret_cast<const VectType*>(dirty_ptr);
  VectType out_vect;

  // Convert to scalar ptr in order to use the []operator
  const auto* psf_scalar   = reinterpret_cast<const ScalarType*>(&psf_vect);
  const auto* dirty_scalar = reinterpret_cast<const ScalarType*>(&dirty_vect);
  auto* out_scalar         = reinterpret_cast<ScalarType*>(&out_vect);

  const auto coeff_gain = coeffs(indices_md[0]) * gain;

  // Apply op
#pragma unroll
  for (int j = 0; j < ITEMS_PER_THREAD; j++) {
    if (j < lanes)
      out_scalar[j] = dirty_scalar[j] - psf_scalar[j] * coeff_gain;
    else
      out_scalar[j] = 0;
  }

  // Store the result
  if (lanes == ITEMS_PER_THREAD)
    *reinterpret_cast<VectType*>(out_ptr) = out_vect;
  else {
    for (int j = 0; j < lanes; j++)
      out_ptr[j] = out_scalar[j];
  }
}

// Type trait mapping VecWidth to CUDA vector type
// template <size_t VecWidth>
// struct cuda_vec;
//
// template <>
// struct cuda_vec<2> { using type = float2; };
//
// template <>
// struct cuda_vec<4> { using type = float4; };
//
// template <size_t VecWidth>
// using cuda_vec_t = typename cuda_vec<VecWidth>::type;

// Row-wise vectorized kernel using float2/float4 loads/stores
// One block per row - handles strided data where stride[2] is not aligned to VecWidth
// but stride[3] == 1 (innermost dimension is contiguous within each row)
// template <size_t VecWidth>
// __global__ void subtract_psf_from_dirty_kernel_row_vec(const float* __restrict__ psf_ptr,
//                                                        const float* __restrict__ dirty_ptr,
//                                                        const float* __restrict__ coeffs,
//                                                        float* __restrict__ out_ptr,
//                                                        float gain,
//                                                        size_t n_channels,
//                                                        size_t n_pol,
//                                                        size_t height,
//                                                        size_t width,
//                                                        size_t stride_ch,
//                                                        size_t stride_pol,
//                                                        size_t stride_row)
// {
//   using vec_t = cuda_vec_t<VecWidth>;
//
//   const size_t row_idx    = blockIdx.x;
//   const size_t total_rows = n_channels * n_pol * height;
//   if (row_idx >= total_rows) return;
//
//   const size_t ch  = row_idx / (n_pol * height);
//   const size_t rem = row_idx % (n_pol * height);
//   const size_t pol = rem / height;
//   const size_t h   = rem % height;
//
//   const size_t row_base = ch * stride_ch + pol * stride_pol + h * stride_row;
//   const float coeff     = coeffs[ch] * gain;
//
//   // Vectorized portion: process VecWidth elements per iteration
//   const size_t vec_width = width / VecWidth;
//   for (size_t vec_i = threadIdx.x; vec_i < vec_width; vec_i += blockDim.x) {
//     const size_t offset = row_base + vec_i * VecWidth;
//
//     const vec_t psf_v   = *reinterpret_cast<const vec_t*>(psf_ptr + offset);
//     const vec_t dirty_v = *reinterpret_cast<const vec_t*>(dirty_ptr + offset);
//
//     vec_t out_v;
//     #pragma unroll
//     for (size_t i = 0; i < VecWidth; ++i) {
//       reinterpret_cast<float*>(&out_v)[i] =
//         reinterpret_cast<const float*>(&dirty_v)[i] -
//         reinterpret_cast<const float*>(&psf_v)[i] * coeff;
//     }
//
//     *reinterpret_cast<vec_t*>(out_ptr + offset) = out_v;
//   }
//
//   // Remainder: handle width % VecWidth elements
//   const size_t remainder_start = vec_width * VecWidth;
//   for (size_t i = remainder_start + threadIdx.x; i < width; i += blockDim.x) {
//     const size_t offset = row_base + i;
//     out_ptr[offset]     = dirty_ptr[offset] - psf_ptr[offset] * coeff;
//   }
// }

// Coalesced scalar kernel - one block per row, threads access consecutive elements
// For when stride[3] == 1 but stride_row is odd (can't use float2/float4)
// __global__ void subtract_psf_from_dirty_kernel_row_coalesced(const float* __restrict__ psf_ptr,
//                                                              const float* __restrict__ dirty_ptr,
//                                                              const float* __restrict__ coeffs,
//                                                              float* __restrict__ out_ptr,
//                                                              float gain,
//                                                              size_t n_channels,
//                                                              size_t n_pol,
//                                                              size_t height,
//                                                              size_t width,
//                                                              size_t stride_ch,
//                                                              size_t stride_pol,
//                                                              size_t stride_row)
// {
//   const size_t row_idx    = blockIdx.x;
//   const size_t total_rows = n_channels * n_pol * height;
//   if (row_idx >= total_rows) return;
//
//   const size_t ch  = row_idx / (n_pol * height);
//   const size_t rem = row_idx % (n_pol * height);
//   const size_t pol = rem / height;
//   const size_t h   = rem % height;
//
//   const size_t row_base = ch * stride_ch + pol * stride_pol + h * stride_row;
//   const float coeff     = coeffs[ch] * gain;
//
//   // Each thread processes elements with stride of blockDim.x
//   // Consecutive threads access consecutive memory addresses (coalesced)
//   for (size_t i = threadIdx.x; i < width; i += blockDim.x) {
//     const size_t offset = row_base + i;
//     out_ptr[offset]     = dirty_ptr[offset] - psf_ptr[offset] * coeff;
//   }
// }

void subtract_psf_from_dirty_async(core::AccessPolicy access_policy,
                                   core::stream_resources& resources,
                                   core::device_span4d_S<float>& psf,
                                   core::device_span4d_S<float>& dirty,
                                   core::device_vect<float>& coeffs,
                                   core::device_span4d_S<float>& out,
                                   float gain)
{
  constexpr int rank   = 4;
  const size_t nb_cols = psf.extent(rank - 1);
  const size_t nb_rows = psf.size() / nb_cols;

  if (true || access_policy.load_policy == core::AccessType::Scalar) {
    subtract_psf_from_dirty_kernel_naive<<<CEIL_DIV(psf.size(), 256), 256, 0, resources.stream>>>(
      psf, dirty, coeffs, out, gain);
  } else if (access_policy.load_policy == core::AccessType::Vec2) {
    constexpr int items_per_thread = 2;
    const size_t nb_vec_cols       = CEIL_DIV(nb_cols, items_per_thread);

    dim3 block(256, 1, 1);
    dim3 grid(CEIL_DIV(nb_vec_cols, block.x),
              static_cast<unsigned int>(nb_rows < 65535 ? nb_rows : 65535),
              static_cast<unsigned int>(CEIL_DIV(nb_rows, nb_rows < 65535 ? nb_rows : 65535)));
    subtract_psf_from_dirty_kernel_vect<float2, float, items_per_thread>
      <<<grid, block, 0, resources.stream>>>(psf, dirty, coeffs, out, gain, nb_rows, nb_cols);
  } else if (access_policy.load_policy == core::AccessType::Vec4) {
    constexpr int items_per_thread = 4;
    const size_t nb_vec_cols       = CEIL_DIV(nb_cols, items_per_thread);

    dim3 block(256, 1, 1);
    dim3 grid(CEIL_DIV(nb_vec_cols, block.x),
              static_cast<unsigned int>(nb_rows < 65535 ? nb_rows : 65535),
              static_cast<unsigned int>(CEIL_DIV(nb_rows, nb_rows < 65535 ? nb_rows : 65535)));
    subtract_psf_from_dirty_kernel_vect<float4, float, items_per_thread>
      <<<grid, block, 0, resources.stream>>>(psf, dirty, coeffs, out, gain, nb_rows, nb_cols);
  }

  CHECK_LAST_CUDA_ERROR();

  // Check if innermost dimension is contiguous (stride[3] == 1) for all spans
  // const bool innermost_contiguous = psf.mapping().stride(3) == 1 &&
  //                                   dirty.mapping().stride(3) == 1 &&
  //                                   out.mapping().stride(3) == 1;
  //
  // // fmt::println("Innermost_contiguous {}", innermost_contiguous);
  //
  //
  // if (!innermost_contiguous || true) {
  //   // Fall back to naive kernel for non-contiguous innermost dimension
  //   subtract_psf_from_dirty_kernel_naive<<<CEIL_DIV(out.size(), 256), 256, 0,
  //   resources.stream>>>(
  //     psf, dirty, coeffs, out, gain);
  //   CHECK_LAST_CUDA_ERROR();
  //   return;
  // }
  //
  // // Innermost is contiguous - use row-wise kernel with appropriate vectorization
  // const size_t n_channels = out.extent(0);
  // const size_t n_pol      = out.extent(1);
  // const size_t height     = out.extent(2);
  // const size_t width      = out.extent(3);
  // const size_t stride_ch  = out.mapping().stride(0);
  // const size_t stride_pol = out.mapping().stride(1);
  // const size_t stride_row = out.mapping().stride(2);
  //
  // const size_t total_rows  = n_channels * n_pol * height;
  // constexpr int block_size = 256;
  //
  // if (stride_row % 4 == 0 && width >= 4) {
  //   fmt::println("float4 vect");
  //   // Best case: float4 vectorization (128-bit loads/stores)
  //   subtract_psf_from_dirty_kernel_row_vec<4><<<total_rows, block_size, 0, resources.stream>>>(
  //     psf.data_handle(), dirty.data_handle(), coeffs.data_handle(), out.data_handle(), gain,
  //     n_channels, n_pol, height, width, stride_ch, stride_pol, stride_row);
  // } else if (stride_row % 2 == 0 && width >= 2) {
  //   fmt::println("float2 vect");
  //   // Good case: float2 vectorization (64-bit loads/stores)
  //   subtract_psf_from_dirty_kernel_row_vec<2><<<total_rows, block_size, 0, resources.stream>>>(
  //     psf.data_handle(), dirty.data_handle(), coeffs.data_handle(), out.data_handle(), gain,
  //     n_channels, n_pol, height, width, stride_ch, stride_pol, stride_row);
  // } else {
  //   fmt::println("fallback coalesced");
  //   // Fallback: coalesced scalar (still better than naive due to coalescing)
  //   subtract_psf_from_dirty_kernel_row_coalesced<<<total_rows, block_size, 0,
  //   resources.stream>>>(
  //     psf.data_handle(), dirty.data_handle(), coeffs.data_handle(), out.data_handle(), gain,
  //     n_channels, n_pol, height, width, stride_ch, stride_pol, stride_row);
  // }
  // CHECK_LAST_CUDA_ERROR();
}

}  // namespace fast_deconv::algo::wscms::detail
