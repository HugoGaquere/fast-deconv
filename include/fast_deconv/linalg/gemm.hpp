#include "cuComplex.h"
#include "cublas_v2.h"

#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cublas_macros.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::linalg {

inline void cgemm_row_major_async(core::stream_resources& resources,
                                  int m,
                                  int n,
                                  int k,
                                  const cuComplex* A,
                                  const cuComplex* B,
                                  cuComplex* C,
                                 cublasOperation_t trans_a = CUBLAS_OP_T,
                                 cublasOperation_t trans_b = CUBLAS_OP_T)
{
  const cuComplex alpha = make_cuFloatComplex(1, 0);
  const cuComplex beta  = make_cuFloatComplex(0, 0);

  CHECK_CUBLAS(cublasCgemm(
    resources.cublas_handle, trans_a, trans_b, n, m, k, &alpha, B, k, A, k, &beta, C, n));
}

inline void gemm_row_major_async(core::stream_resources& resources,
                                 int m,
                                 int n,
                                 int k,
                                 const float* A,
                                 const float* B,
                                 float* C,
                                 cublasOperation_t trans_a = CUBLAS_OP_T,
                                 cublasOperation_t trans_b = CUBLAS_OP_T)
{
  const float alpha{1.f};
  const float beta{0.f};

  CHECK_CUBLAS(cublasSgemm(
    resources.cublas_handle, trans_a, trans_b, n, m, k, &alpha, B, k, A, k, &beta, C, n));
}

}  // namespace fast_deconv::linalg
