#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::linalg {

// Maximum supported column count for the per-thread solve arrays.
static constexpr int PINV_MAX_COLS = 16;

namespace detail {

// Pseudo-inverse via normal equations: A+ = (A^T A)^{-1} A^T
// Uses Cholesky factorization for the N×N system (A^T A is SPD for full column rank A).
// A is M×N (row-major), A_pinv is N×M (row-major). Requires M >= N.
// Single-block kernel; all computation in shared memory.
//
// Dynamic shared memory layout (floats):
//   As  : M * N   — copy of A
//   AtA : N * N   — A^T A
//   At  : N * M   — A^T
//   L   : N * N   — Cholesky factor (lower triangular)
static __global__ void pinv_kernel(const float* __restrict__ A,
                            float* __restrict__ A_pinv,
                            int M,
                            int N)
{
  extern __shared__ float smem[];

  float* As  = smem;
  float* AtA = As  + M * N;
  float* At  = AtA + N * N;
  float* L   = At  + N * M;

  const int tid = threadIdx.x;

  // ---------------------------------------------------------------
  // 1. Load A into shared memory
  // ---------------------------------------------------------------
  for (int i = tid; i < M * N; i += blockDim.x)
    As[i] = A[i];

  __syncthreads();

  // ---------------------------------------------------------------
  // 2. AtA = A^T * A  (N×N, symmetric)
  //    AtA(i,j) = sum_k A(k,i) * A(k,j)
  // ---------------------------------------------------------------
  for (int idx = tid; idx < N * N; idx += blockDim.x) {
    const int i = idx / N;
    const int j = idx % N;
    float sum = 0.0f;
    for (int k = 0; k < M; ++k)
      sum += As[k * N + i] * As[k * N + j];
    AtA[i * N + j] = sum;
  }

  __syncthreads();

  // ---------------------------------------------------------------
  // 3. At = A^T  (N×M)
  //    At(i,j) = A(j,i)
  // ---------------------------------------------------------------
  for (int idx = tid; idx < N * M; idx += blockDim.x) {
    const int i = idx / M;
    const int j = idx % M;
    At[i * M + j] = As[j * N + i];
  }

  __syncthreads();

  // ---------------------------------------------------------------
  // 4. Cholesky factorization: AtA = L * L^T  (serial, N is small)
  // ---------------------------------------------------------------
  if (tid == 0) {
    for (int i = 0; i < N; ++i) {
      // Zero upper triangle of L row
      for (int j = i + 1; j < N; ++j)
        L[i * N + j] = 0.0f;

      for (int j = 0; j <= i; ++j) {
        float sum = AtA[i * N + j];
        for (int k = 0; k < j; ++k)
          sum -= L[i * N + k] * L[j * N + k];
        if (i == j)
          L[i * N + j] = sqrtf(sum);
        else
          L[i * N + j] = sum / L[j * N + j];
      }
    }
  }

  __syncthreads();

  // ---------------------------------------------------------------
  // 5. Solve L L^T X = A^T  for X = A+  (N×M)
  //    Each thread handles one or more columns of the RHS.
  // ---------------------------------------------------------------
  for (int col = tid; col < M; col += blockDim.x) {
    // Forward substitution: L y = At[:, col]
    float y[PINV_MAX_COLS];
    for (int i = 0; i < N; ++i) {
      float sum = At[i * M + col];
      for (int k = 0; k < i; ++k)
        sum -= L[i * N + k] * y[k];
      y[i] = sum / L[i * N + i];
    }

    // Back substitution: L^T x = y
    float x[PINV_MAX_COLS];
    for (int i = N - 1; i >= 0; --i) {
      float sum = y[i];
      for (int k = i + 1; k < N; ++k)
        sum -= L[k * N + i] * x[k];
      x[i] = sum / L[i * N + i];
    }

    // Write result to A_pinv[:, col]
    for (int i = 0; i < N; ++i)
      A_pinv[i * M + col] = x[i];
  }
}

}  // namespace detail

// Host launcher for the pseudo-inverse kernel.
// Computes A+ = (A^T A)^{-1} A^T via Cholesky factorization.
// A is (rows x cols) row-major, A_pinv is (cols x rows) row-major.
// Requires rows >= cols and A to have full column rank.
inline void pinv(const core::device_span2d<float>& A,
                 core::device_span2d<float>& A_pinv,
                 int rows,
                 int cols,
                 core::stream_resources& resources)
{
  assert(rows >= cols && "pinv requires rows >= cols (overdetermined system)");
  assert(cols <= PINV_MAX_COLS && "cols exceeds PINV_MAX_COLS for per-thread solve arrays");

  const int block_size = (rows > 256) ? 256 : ((rows + 31) / 32) * 32;

  // Dynamic shared memory: As(M*N) + AtA(N*N) + At(N*M) + L(N*N)
  const size_t smem_bytes = static_cast<size_t>(
    rows * cols + cols * cols + cols * rows + cols * cols) * sizeof(float);

  detail::pinv_kernel<<<1, block_size, smem_bytes, resources.stream>>>(
    A.data_handle(), A_pinv.data_handle(), rows, cols);
  CHECK_LAST_CUDA_ERROR();
}

}  // namespace fast_deconv::linalg
