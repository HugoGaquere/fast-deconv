#pragma once

#include <cub/cub.cuh>

#include <emu/cuda/device/mdspan.hpp>
#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fast_deconv/util/mdspan_utils.hpp>

namespace cpts = fast_deconv::core::cpts;

namespace {

template <cpts::mdspan Mdspan>
__device__ inline auto linear_to_indices(typename Mdspan::size_type lin, const Mdspan& m)
{
  using index_t   = typename Mdspan::index_type;
  constexpr int R = Mdspan::rank();

  std::array<index_t, R> idx{};
#pragma unroll
  for (int d = R - 1; d >= 0; --d) {
    const auto e = static_cast<index_t>(m.extent(d));
    idx[d]       = static_cast<index_t>(lin % e);
    lin          = static_cast<typename Mdspan::size_type>(lin / e);
  }
  return idx;
}

template <typename Mdspan, std::size_t... Is>
__device__ inline decltype(auto) at_impl(
  Mdspan&& m,
  const std::array<typename std::remove_reference_t<Mdspan>::index_type,
                   std::remove_reference_t<Mdspan>::rank()>& idx,
  std::index_sequence<Is...>)
{
  return m(idx[Is]...);
}

template <typename Mdspan>
__device__ inline decltype(auto) at(
  Mdspan&& m,
  const std::array<typename std::remove_reference_t<Mdspan>::index_type,
                   std::remove_reference_t<Mdspan>::rank()>& idx)
{
  using M = std::remove_reference_t<Mdspan>;
  return at_impl(std::forward<Mdspan>(m), idx, std::make_index_sequence<M::rank()>{});
}

template <cpts::mdspan Mdspan>
__global__ void simple_subtract_kernel(const Mdspan A, const Mdspan B, Mdspan C)
{
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= A.size()) return;
  auto idx   = linear_to_indices<Mdspan>(tid, A);
  at(C, idx) = at(A, idx) - at(B, idx);
}

__global__ void subtract_kernel_vect_load(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          size_t size)
{
  const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  const auto* a = reinterpret_cast<const float4*>(A);
  const auto* b = reinterpret_cast<const float4*>(B);
  auto* c       = reinterpret_cast<float4*>(C);

  for (int i = tid; i < size / 4; i += stride) {
    float4 va = a[i];
    float4 vb = b[i];
    float4 vc;
    vc.x = va.x - vb.x;
    vc.y = va.y - vb.y;
    vc.z = va.z - vb.z;
    vc.w = va.w - vb.w;

    c[i] = vc;
  }

  // in only one thread, process final elements (if there are any)
  int remainder = size % 4;
  if (tid == 0 && remainder != 0) {
    while (remainder) {
      int idx = size - remainder--;
      C[idx]  = A[idx] - B[idx];
    }
  }
}

template <typename T,
          int BLOCK_THREADS,
          int ITEMS_PER_THREAD,
          cub::BlockLoadAlgorithm LOAD_ALGORITHM,
          cub::BlockStoreAlgorithm STORE_ALGORITHM>
__global__ void subtract_kernel_cub_load(const T* __restrict__ A,
                                         const T* __restrict__ B,
                                         T* __restrict__ C,
                                         size_t size)
{
  using block_load  = cub::BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM>;
  using block_store = cub::BlockStore<T, BLOCK_THREADS, ITEMS_PER_THREAD, STORE_ALGORITHM>;

  // Obtain this block's segment of consecutive items
  T thread_a[ITEMS_PER_THREAD];
  T thread_b[ITEMS_PER_THREAD];
  T thread_c[ITEMS_PER_THREAD];
  int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);

  block_load().Load(A + block_offset, thread_a);
  block_load().Load(B + block_offset, thread_b);

// Do the subtraction
#pragma unroll ITEMS_PER_THREAD
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    thread_c[i] = thread_a[i] - thread_b[i];
  }

  // Store back to C
  block_store().Store(C + block_offset, thread_c);
}

}  // namespace

namespace fast_deconv::matrix::detail {

template <cpts::mdspan Mdspan>
requires cpts::is_layout_stride<Mdspan>  // or cpts::is_layout_right<Mdspan>
  void subtract_async(const Mdspan& A,
                      const Mdspan& B,
                      Mdspan& C,
                      core::stream_resources& resources)
{
  constexpr std::size_t vec_bytes = 16;  // float4

  bool are_all_inner_unit_stride = util::all_inner_unit_stride(A, B, C);
  bool are_all_aligned_for_vec   = util::all_aligned_for_vec(vec_bytes, A, B, C);
  fmt::println("unit_stride: {}, aligned: {}", are_all_inner_unit_stride, are_all_aligned_for_vec);

  if (!util::all_inner_unit_stride(A, B, C)) {
    simple_subtract_kernel<<<CEIL_DIV(A.size(), 256), 256, 0, resources.stream>>>(A, B, C);
    CHECK_LAST_CUDA_ERROR();
    return;
  }

  if(util::all_aligned_for_vec(16, A, B, C)) {
    // run with 4 float vect
  }
  else if(util::all_aligned_for_vec(8, A, B, C)) {
    // run with 2 float vect
  } else {
    // 1 float
  }

}

template <emu::cuda::device::cpts::mdspan Mdspan>
requires cpts::is_layout_right<Mdspan> void subtract_async(const Mdspan& A,
                                                           const Mdspan& B,
                                                           Mdspan& C,
                                                           core::stream_resources& resources)
{
  const auto stream          = resources.stream;
  const size_t size          = A.size();
  const int items_per_thread = 4;
  const int block_dim        = 128;
  const int grid_dim         = CEIL_DIV(size / items_per_thread, block_dim);

  subtract_kernel_cub_load<float,
                           block_dim,
                           items_per_thread,
                           cub::BLOCK_LOAD_VECTORIZE,
                           cub::BLOCK_STORE_VECTORIZE>
    <<<grid_dim, block_dim, 0, stream>>>(A.data_handle(), B.data_handle(), C.data_handle(), size);
  CHECK_LAST_CUDA_ERROR();
}

template <cpts::mdspan Mdspan>
requires cpts::is_layout_left<Mdspan> void subtract_async(const Mdspan& A,
                                                          const Mdspan& B,
                                                          Mdspan& C,
                                                          core::stream_resources& resources)
{
  static_assert(!cpts::is_layout_left<Mdspan>,
                "subtract_async: layout_left is not implemented yet.");
}

// void subtract_async(
//   const float* A, const float* B, float* C, size_t size, core::stream_resources& resources)
// {
//   const auto stream          = resources.stream;
//   const int items_per_thread = 4;
//   const int block_dim        = 128;
//   const int grid_dim         = CEIL_DIV(size / items_per_thread, block_dim);
//
//   subtract_kernel_cub_load<float,
//                            block_dim,
//                            items_per_thread,
//                            cub::BLOCK_LOAD_VECTORIZE,
//                            cub::BLOCK_STORE_VECTORIZE>
//     <<<grid_dim, block_dim, 0, stream>>>(A, B, C, size);
//
//   CHECK_LAST_CUDA_ERROR();
// }

}  // namespace fast_deconv::matrix::detail
