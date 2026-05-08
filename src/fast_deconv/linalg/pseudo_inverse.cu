#include <fast_deconv/linalg/pseudo_inverse.hpp>
#include <fast_deconv/core/logger.hpp>


namespace fast_deconv::linalg {

void compute_pseudo_inverse(const core::stream_resources& stream_res, const float* d_A,
                            float* d_Apinv, int n_rows, int n_cols)
{
  const auto handle = stream_res.cublas_handle;
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  const bool underdetermined = n_cols > n_rows;
  const int g_dim = underdetermined ? n_rows : n_cols;

  float* d_G = stream_res.alloc_async<float>(g_dim * g_dim);
  float* d_Ginv = stream_res.alloc_async<float>(g_dim * g_dim);
  int* d_info = stream_res.alloc_async<int>(1);
  float** d_G_ptrs = stream_res.alloc_async<float*>(1);
  float** d_Ginv_ptrs = stream_res.alloc_async<float*>(1);

  // Step 1: G
  //   overdetermined: G = A_cm @ A_cm^T  (N, T)  -> A^T A   [n_cols, n_cols]
  //   underdetermined: G = A_cm^T @ A_cm (T, N)  -> A A^T   [n_rows,  n_rows]
  if (underdetermined) {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n_rows, n_rows, n_cols, &alpha,
                             d_A, n_cols, d_A, n_cols, &beta, d_G, n_rows));
  } else {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n_cols, n_cols, n_rows, &alpha,
                             d_A, n_cols, d_A, n_cols, &beta, d_G, n_cols));
  }

  // Step 2: G_inv = inv(G)
  // matinvBatched expects device arrays of device pointers;
  // copy host-side pointer values to device so cuBLAS can read them
  CHECK_CUDA(cudaMemcpyAsync(d_G_ptrs, &d_G, sizeof(float*), cudaMemcpyHostToDevice,
                             stream_res.cuda_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_Ginv_ptrs, &d_Ginv, sizeof(float*), cudaMemcpyHostToDevice,
                             stream_res.cuda_stream));
  CHECK_CUBLAS(
      cublasSmatinvBatched(handle, g_dim, d_G_ptrs, g_dim, d_Ginv_ptrs, g_dim, d_info, 1));

#if FD_LOG_ACTIVE_LEVEL <= FD_LOG_LEVEL_DEBUG
  {
    int h_info = 0;
    CHECK_CUDA(cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost,
                               stream_res.cuda_stream));
    stream_res.sync();
    FD_LOG_DEBUG(
        "compute_pseudo_inverse: n_rows={} n_cols={} mode={} g_dim={} matinvBatched info={} ({})",
        n_rows, n_cols, underdetermined ? "underdetermined" : "overdetermined", g_dim, h_info,
        h_info == 0 ? "ok" : "singular");
  }
#endif

  // Step 3: A_pinv [n_cols, n_rows] col-major
  //   overdetermined: A_pinv_cm = G_inv @ A_cm         -> inv(A^T A) @ A^T
  //   underdetermined: A_pinv_cm = A_cm  @ G_inv       -> A^T @ inv(A A^T)
  if (underdetermined) {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_cols, n_rows, n_rows, &alpha,
                             d_A, n_cols, d_Ginv, n_rows, &beta, d_Apinv, n_cols));
  } else {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_cols, n_rows, n_cols, &alpha,
                             d_Ginv, n_cols, d_A, n_cols, &beta, d_Apinv, n_cols));
  }

  stream_res.free_async(d_Ginv_ptrs);
  stream_res.free_async(d_G_ptrs);
  stream_res.free_async(d_info);
  stream_res.free_async(d_Ginv);
  stream_res.free_async(d_G);
}


}  // namespace fast_deconv::multi_frequency