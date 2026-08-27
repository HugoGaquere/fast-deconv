#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/linalg/pseudo_inverse.hpp>

namespace fast_deconv::linalg {

void compute_pseudo_inverse(const core::exec_ctx& ctx, const float* d_A, float* d_Apinv, int n_rows, int n_cols)
{
  const auto handle = ctx.cublas_handle;
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  const bool underdetermined = n_cols > n_rows;
  const int g_dim = underdetermined ? n_rows : n_cols;

  const auto G = ctx.alloc_ptr_async<float>(g_dim * g_dim);
  const auto Ginv = ctx.alloc_ptr_async<float>(g_dim * g_dim);
  const auto info = ctx.alloc_ptr_async<int>(1);
  const auto G_ptrs = ctx.alloc_ptr_async<float*>(1);
  const auto Ginv_ptrs = ctx.alloc_ptr_async<float*>(1);

  // Held as lvalues: matinvBatched reads their addresses below.
  float* const d_G = G.get();
  float* const d_Ginv = Ginv.get();

  // Step 1: G
  //   overdetermined: G = A_cm @ A_cm^T  (N, T)  -> A^T A   [n_cols, n_cols]
  //   underdetermined: G = A_cm^T @ A_cm (T, N)  -> A A^T   [n_rows,  n_rows]
  if (underdetermined) {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n_rows, n_rows, n_cols, &alpha, d_A, n_cols, d_A, n_cols,
                             &beta, d_G, n_rows));
  } else {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n_cols, n_cols, n_rows, &alpha, d_A, n_cols, d_A, n_cols,
                             &beta, d_G, n_cols));
  }

  // Step 2: G_inv = inv(G)
  // matinvBatched expects device arrays of device pointers;
  // copy host-side pointer values to device so cuBLAS can read them
  CHECK_CUDA(cudaMemcpyAsync(G_ptrs.get(), &d_G, sizeof(float*), cudaMemcpyHostToDevice, ctx.cuda_stream));
  CHECK_CUDA(cudaMemcpyAsync(Ginv_ptrs.get(), &d_Ginv, sizeof(float*), cudaMemcpyHostToDevice, ctx.cuda_stream));
  CHECK_CUBLAS(cublasSmatinvBatched(handle, g_dim, G_ptrs.get(), g_dim, Ginv_ptrs.get(), g_dim, info.get(), 1));

#if FD_LOG_ACTIVE_LEVEL <= FD_LOG_LEVEL_DEBUG
  {
    int h_info = 0;
    CHECK_CUDA(cudaMemcpyAsync(&h_info, info.get(), sizeof(int), cudaMemcpyDeviceToHost, ctx.cuda_stream));
    ctx.wait();
    FD_LOG_DEBUG("compute_pseudo_inverse: n_rows={} n_cols={} mode={} g_dim={} matinvBatched info={} ({})", n_rows,
                 n_cols, underdetermined ? "underdetermined" : "overdetermined", g_dim, h_info,
                 h_info == 0 ? "ok" : "singular");
  }
#endif

  // Step 3: A_pinv [n_cols, n_rows] col-major
  //   overdetermined: A_pinv_cm = G_inv @ A_cm         -> inv(A^T A) @ A^T
  //   underdetermined: A_pinv_cm = A_cm  @ G_inv       -> A^T @ inv(A A^T)
  if (underdetermined) {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_cols, n_rows, n_rows, &alpha, d_A, n_cols, d_Ginv,
                             n_rows, &beta, d_Apinv, n_cols));
  } else {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_cols, n_rows, n_cols, &alpha, d_Ginv, n_cols, d_A,
                             n_cols, &beta, d_Apinv, n_cols));
  }
}

}  // namespace fast_deconv::linalg