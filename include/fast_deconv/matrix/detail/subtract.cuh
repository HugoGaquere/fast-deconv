#pragma once

#include <cub/cub.cuh>

#include <emu/cuda/device/mdspan.hpp>
#include <fast_deconv/core/access_policy.hpp>
#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fast_deconv/util/mdspan_utils.hpp>

#include <tuple>

namespace {

namespace cpts = fast_deconv::core::cpts;

template <cpts::mdspan Mdspan>
__device__ __forceinline__ auto linear_to_indices(typename Mdspan::size_type lin, const Mdspan& m)
{
  using size_t_   = typename Mdspan::size_type;
  using index_t   = typename Mdspan::index_type;
  constexpr int R = Mdspan::rank();

  std::array<index_t, R> idx{};

#pragma unroll
  for (int d = R - 1; d >= 0; --d) {
    const auto e    = static_cast<size_t_>(m.extent(d));
    const size_t_ r = lin % e;
    lin /= e;
    idx[d] = static_cast<index_t>(r);
  }
  return idx;
}

template <typename Mdspan, std::size_t... Is>
__device__ __forceinline__ decltype(auto) at_impl(
  Mdspan&& m,
  const std::array<typename std::remove_reference_t<Mdspan>::index_type,
                   std::remove_reference_t<Mdspan>::rank()>& idx,
  std::index_sequence<Is...>)
{
  return m(idx[Is]...);
}

template <typename Mdspan>
__device__ __forceinline__ decltype(auto) at(
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

template <typename ItemType, int ITEMS_PER_THREAD>
__global__ void subtract_kernel_vect_load(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          size_t size)
{
  const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  const auto* a = reinterpret_cast<const ItemType*>(A);
  const auto* b = reinterpret_cast<const ItemType*>(B);
  auto* c       = reinterpret_cast<ItemType*>(C);

  for (int i = tid; i < size / ITEMS_PER_THREAD; i += stride) {
    ItemType va = a[i];
    ItemType vb = b[i];
    ItemType vc;
#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
      reinterpret_cast<float*>(&vc)[j] =
        reinterpret_cast<const float*>(&va)[j] - reinterpret_cast<const float*>(&vb)[j];
    }
    c[i] = vc;
  }

  // in only one thread, process final elements (if there are any)
  int remainder = size % ITEMS_PER_THREAD;
  if (tid == 0 && remainder != 0) {
    while (remainder) {
      int idx = size - remainder--;
      C[idx]  = A[idx] - B[idx];
    }
  }
}

template <cpts::mdspan Mdspan>
__device__ __forceinline__ auto item_offset(uint idx, const Mdspan& m)
{
  const auto tid_md = linear_to_indices(idx, m);
  const auto offset = std::apply([&m](auto... i) { return m.mapping()(i...); }, tid_md);
  return offset;
}

template <typename VectType, cpts::mdspan Mdspan>
__device__ __forceinline__ VectType load(uint idx, const Mdspan& m)
{
  // using item_type = Mdspan::element_type;
  const auto offset = item_offset(idx, m);
  const auto* ptr   = m.data_handle() + offset;
  return *reinterpret_cast<const VectType*>(ptr);
  // const auto m_vect = *reinterpret_cast<const VectType*>(ptr);
  // return m_vect;
}

template <typename VectType, typename ScalarType, int ITEMS_PER_THREAD, cpts::mdspan Mdspan>
__global__ void subtract_kernel_vect_load_stride_2d(
  const Mdspan A, const Mdspan B, Mdspan C, uint nb_rows, uint nb_cols)
{
  const size_t row = blockIdx.y + gridDim.y * blockIdx.z;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x * ITEMS_PER_THREAD;
  if (row >= nb_rows || col >= nb_cols) return;

  const int rem_in_row = static_cast<int>(nb_cols - col);
  const int lanes      = min(ITEMS_PER_THREAD, rem_in_row);
  const size_t lin     = row * nb_cols + col;

  // Compute indices once
  const auto tid_md = linear_to_indices(lin, A);

  // Compute all pointers upfront
  const auto* a_ptr =
    A.data_handle() + std::apply([&](auto... i) { return A.mapping()(i...); }, tid_md);
  const auto* b_ptr =
    B.data_handle() + std::apply([&](auto... i) { return B.mapping()(i...); }, tid_md);
  auto* c_ptr = C.data_handle() + std::apply([&](auto... i) { return C.mapping()(i...); }, tid_md);

  // Load vect
  const auto a_vect = *reinterpret_cast<const VectType*>(a_ptr);
  const auto b_vect = *reinterpret_cast<const VectType*>(b_ptr);
  VectType c_vect;

  // Convert to scalar ptr in order to use the []operator
  const auto* a_scalar = reinterpret_cast<const ScalarType*>(&a_vect);
  const auto* b_scalar = reinterpret_cast<const ScalarType*>(&b_vect);
  auto* c_scalar       = reinterpret_cast<ScalarType*>(&c_vect);

  // Apply op
#pragma unroll
  for (int j = 0; j < ITEMS_PER_THREAD; j++) {
    if (j < lanes)
      c_scalar[j] = a_scalar[j] - b_scalar[j];
    else
      c_scalar[j] = 0;
  }

  // Store the result
  if (lanes == ITEMS_PER_THREAD)
    *reinterpret_cast<VectType*>(c_ptr) = c_vect;
  else {
    for (int j = 0; j < lanes; j++)
      c_ptr[j] = c_scalar[j];
  }

  // OLD

  // const size_t lin     = row * nb_cols + col;
  // const auto tid_md = linear_to_indices(lin, A);
  //
  // const auto a_offset = std::apply([&](auto... i) { return A.mapping()(i...); }, tid_md);
  // const auto* a_ptr   = A.data_handle() + a_offset;
  //
  // const auto b_offset = std::apply([&](auto... i) { return B.mapping()(i...); }, tid_md);
  // const auto* b_ptr   = B.data_handle() + b_offset;
  //
  // const auto c_offset = std::apply([&](auto... i) { return C.mapping()(i...); }, tid_md);
  // auto* c_ptr         = C.data_handle() + c_offset;
  //
  // const auto a_vect = *reinterpret_cast<const VectType*>(a_ptr);
  // const auto b_vect = *reinterpret_cast<const VectType*>(b_ptr);
  // VectType c_vect;
  //
  // const auto* a_scalar = reinterpret_cast<const ScalarType*>(&a_vect);
  // const auto* b_scalar = reinterpret_cast<const ScalarType*>(&b_vect);
  // auto* c_scalar       = reinterpret_cast<ScalarType*>(&c_vect);
  //
  //   int rem_in_row = int(nb_cols - col);
  //   int lanes      = min(ITEMS_PER_THREAD, rem_in_row);
  //
  // #pragma unroll
  //   for (int j = 0; j < ITEMS_PER_THREAD; j++) {
  //     if (j < lanes)
  //       c_scalar[j] = a_scalar[j] - b_scalar[j];
  //     else
  //       c_scalar[j] = 0;  // won't be stored
  //   }
  //
  //   // masked store
  //   if (lanes == ITEMS_PER_THREAD)
  //     *reinterpret_cast<VectType*>(c_ptr) = c_vect;
  //   else {
  //     for (int j = 0; j < lanes; j++)
  //       c_ptr[j] = c_scalar[j];
  //   }
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

template <core::cpts::mdspan Mdspan>
void subtract_async(core::AccessPolicy access_policy,
                    core::stream_resources& resources,
                    const Mdspan& A,
                    const Mdspan& B,
                    Mdspan& C)
{
  static_assert(!core::cpts::is_layout_left<Mdspan>,
                "subtract_async: layout_left is not implemented yet.");

  constexpr int rank   = Mdspan::rank();
  const size_t nb_cols = A.extent(rank - 1);
  const size_t nb_rows = A.size() / nb_cols;

  if (access_policy.load_policy == core::AccessType::Scalar) {
    simple_subtract_kernel<<<CEIL_DIV(A.size(), 256), 256, 0, resources.stream>>>(A, B, C);
  } else if (access_policy.load_policy == core::AccessType::Vec2) {
    constexpr int items_per_thread = 2;
    const size_t nb_vec_cols       = CEIL_DIV(nb_cols, items_per_thread);

    dim3 block(256, 1, 1);
    dim3 grid(CEIL_DIV(nb_vec_cols, block.x),
              static_cast<unsigned int>(nb_rows < 65535 ? nb_rows : 65535),
              static_cast<unsigned int>(CEIL_DIV(nb_rows, nb_rows < 65535 ? nb_rows : 65535)));
    subtract_kernel_vect_load_stride_2d<float2, float, items_per_thread>
      <<<grid, block, 0, resources.stream>>>(A, B, C, nb_rows, nb_cols);
  } else if (access_policy.load_policy == core::AccessType::Vec4) {
    constexpr int items_per_thread = 4;
    const size_t nb_vec_cols       = CEIL_DIV(nb_cols, items_per_thread);

    dim3 block(256, 1, 1);
    dim3 grid(CEIL_DIV(nb_vec_cols, block.x),
              static_cast<unsigned int>(nb_rows < 65535 ? nb_rows : 65535),
              static_cast<unsigned int>(CEIL_DIV(nb_rows, nb_rows < 65535 ? nb_rows : 65535)));
    subtract_kernel_vect_load_stride_2d<float4, float, items_per_thread>
      <<<grid, block, 0, resources.stream>>>(A, B, C, nb_rows, nb_cols);
  }

  CHECK_LAST_CUDA_ERROR();
}

// template <cpts::mdspan Mdspan>
// requires cpts::is_layout_stride<Mdspan> void subtract_async(core::stream_resources& resources,
//                                                             const Mdspan& A,
//                                                             const Mdspan& B,
//                                                             Mdspan& C)
// {
//   simple_subtract_kernel<<<CEIL_DIV(A.size(), 256), 256, 0, resources.stream>>>(A, B, C);
//   CHECK_LAST_CUDA_ERROR();
// }
//
// template <cpts::mdspan Mdspan>
// requires cpts::is_layout_right<Mdspan> void subtract_async(core::stream_resources& resources,
//                                                            const Mdspan& A,
//                                                            const Mdspan& B,
//                                                            Mdspan& C)
// {
//   const auto stream          = resources.stream;
//   const size_t size          = A.size();
//   const int items_per_thread = 4;
//   const int block_dim        = 128;
//   const int grid_dim         = CEIL_DIV(size / items_per_thread, block_dim);
//
//   subtract_kernel_cub_load<float,
//                            block_dim,
//                            items_per_thread,
//                            cub::BLOCK_LOAD_VECTORIZE,
//                            cub::BLOCK_STORE_VECTORIZE>
//     <<<grid_dim, block_dim, 0, stream>>>(A.data_handle(), B.data_handle(), C.data_handle(),
//     size);
//   CHECK_LAST_CUDA_ERROR();
// }
//
// template <cpts::mdspan Mdspan>
// requires cpts::is_layout_left<Mdspan> void subtract_async(core::stream_resources& resources,
//                                                           const Mdspan& A,
//                                                           const Mdspan& B,
//                                                           Mdspan& C)
// {
//   static_assert(!cpts::is_layout_left<Mdspan>,
//                 "subtract_async: layout_left is not implemented yet.");
// }

}  // namespace fast_deconv::matrix::detail
