// #include "kronecker_mixed_product_example.cuh"
//
// #include <fast_deconv/core/mdarray.hpp>
// #include <fast_deconv/core/stream_resources.hpp>
// #include <fast_deconv/linalg/gemm.hpp>
// #include <fast_deconv/linalg/kronecker.cuh>
// #include <fast_deconv/util/cuda_macros.hpp>
// #include <fast_deconv/util/kernel_bench.hpp>
//
// #include <cmath>
// #include <cstdio>
// #include <limits>
// #include <vector>
//
// template <typename T>
// void fill_with_random(T arr)
// {
//   for (int i = 0; i < arr.size(); i++) {
//     arr.data_handle()[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
//   }
// }
//
// #include <cmath>
// #include <cstdio>
// #include <limits>
// #include <vector>
//
// // Simple row-major index helper
// inline std::size_t idx(std::size_t i, std::size_t j, std::size_t ncols) { return i * ncols + j; }
//
// // Plain CPU GEMM (row-major): C[m,n] = A[m,k] * B[k,n]
// void gemm_row_major_cpu(
//   const float* A, const float* B, float* C, std::size_t m, std::size_t n, std::size_t k)
// {
//   for (std::size_t i = 0; i < m; ++i) {
//     for (std::size_t j = 0; j < n; ++j) {
//       double acc = 0.0;  // use double for accumulation, store back as float
//       for (std::size_t kk = 0; kk < k; ++kk) {
//         acc += static_cast<double>(A[idx(i, kk, k)]) * B[idx(kk, j, n)];
//       }
//       C[idx(i, j, n)] = static_cast<float>(acc);
//     }
//   }
// }
//
// // CPU Kronecker for mixed-product: res[(a*b) x (c*d)] = kron(AC[a,c], BD[b,d])
// void kronecker_mixed_product_cpu(const float* AC,
//                                  const float* BD,
//                                  float* res,
//                                  std::size_t a_rows,
//                                  std::size_t c_cols,
//                                  std::size_t b_rows,
//                                  std::size_t d_cols)
// {
//   const std::size_t res_rows = a_rows * b_rows;
//   const std::size_t res_cols = c_cols * d_cols;
//
//   for (std::size_t iA = 0; iA < a_rows; ++iA) {
//     for (std::size_t jA = 0; jA < c_cols; ++jA) {
//       const float ac_val = AC[idx(iA, jA, c_cols)];
//
//       for (std::size_t iB = 0; iB < b_rows; ++iB) {
//         for (std::size_t jB = 0; jB < d_cols; ++jB) {
//           const float bd_val = BD[idx(iB, jB, d_cols)];
//
//           const std::size_t i_res = iA * b_rows + iB;
//           const std::size_t j_res = jA * d_cols + jB;
//
//           res[idx(i_res, j_res, res_cols)] = ac_val * bd_val;
//         }
//       }
//     }
//   }
// }
//
// void run_kronecker_mixed_product_example()
// {
//   std::printf("Running kronecker Mixed Product example\n");
//
//   using uint = unsigned int;
//
//   uint a_rows = 10;
//   uint a_cols = 10;
//   uint b_rows = 601;
//   uint b_cols = 601;
//   uint c_rows = 10;
//   uint c_cols = 20;
//   uint d_rows = 601;
//   uint d_cols = 61;
//
//   auto A = fast_deconv::core::make_managed_mdarray<float>(a_rows, a_cols);
//   auto B = fast_deconv::core::make_managed_mdarray<float>(b_rows, b_cols);
//   auto C = fast_deconv::core::make_managed_mdarray<float>(c_rows, c_cols);
//   auto D = fast_deconv::core::make_managed_mdarray<float>(d_rows, d_cols);
//
//   auto AC = fast_deconv::core::make_managed_mdarray<float>(a_rows, c_cols);
//   auto BD = fast_deconv::core::make_managed_mdarray<float>(b_rows, d_cols);
//
//   auto res = fast_deconv::core::make_managed_mdarray<float>(a_rows * b_rows, c_cols * d_cols);
//
//   fill_with_random(A.view());
//   fill_with_random(B.view());
//   fill_with_random(C.view());
//   fill_with_random(D.view());
//
//   fast_deconv::core::stream_resources resources;
//
//   // ---- GPU path (your existing code) ----
//   fast_deconv::util::run_benchmark(
//     [&]() -> void {
//       fast_deconv::linalg::gemm_row_major_async(resources,
//                                                 a_rows,
//                                                 c_cols,
//                                                 a_cols,
//                                                 A.view().data_handle(),
//                                                 C.view().data_handle(),
//                                                 AC.view().data_handle());
//
//       fast_deconv::linalg::gemm_row_major_async(resources,
//                                                 b_rows,
//                                                 d_cols,
//                                                 b_cols,
//                                                 B.view().data_handle(),
//                                                 D.view().data_handle(),
//                                                 BD.view().data_handle());
//
//       fast_deconv::linalg::kronecker_async(resources,
//                                            AC.view().data_handle(),
//                                            BD.view().data_handle(),
//                                            res.view().data_handle(),
//                                            a_rows,
//                                            c_cols,
//                                            b_rows,
//                                            d_cols);
//     },
//     resources.stream);
//
//   // Make sure GPU work is done before reading managed memory on CPU
//   resources.sync();
//
//   // ---- CPU reference computation ----
//
//   const std::size_t a_rows_s = a_rows;
//   const std::size_t a_cols_s = a_cols;
//   const std::size_t b_rows_s = b_rows;
//   const std::size_t b_cols_s = b_cols;
//   const std::size_t c_rows_s = c_rows;
//   const std::size_t c_cols_s = c_cols;
//   const std::size_t d_rows_s = d_rows;
//   const std::size_t d_cols_s = d_cols;
//
//   // Allocate CPU buffers for AC_ref, BD_ref, res_ref
//   std::vector<float> AC_ref(a_rows_s * c_cols_s);
//   std::vector<float> BD_ref(b_rows_s * d_cols_s);
//   std::vector<float> res_ref(a_rows_s * b_rows_s * c_cols_s * d_cols_s);
//
//   const float* A_ptr   = A.view().data_handle();
//   const float* B_ptr   = B.view().data_handle();
//   const float* C_ptr   = C.view().data_handle();
//   const float* D_ptr   = D.view().data_handle();
//   const float* res_gpu = res.view().data_handle();
//
//   // (AC)_ref = A * C
//   gemm_row_major_cpu(A_ptr, C_ptr, AC_ref.data(), a_rows_s, c_cols_s, a_cols_s);
//
//   // (BD)_ref = B * D
//   gemm_row_major_cpu(B_ptr, D_ptr, BD_ref.data(), b_rows_s, d_cols_s, b_cols_s);
//
//   // res_ref = kron(AC_ref, BD_ref)
//   kronecker_mixed_product_cpu(
//     AC_ref.data(), BD_ref.data(), res_ref.data(), a_rows_s, c_cols_s, b_rows_s, d_cols_s);
//
//   // ---- Compare GPU result vs CPU reference ----
//   const std::size_t res_size = a_rows_s * b_rows_s * c_cols_s * d_cols_s;
//
//   double max_abs_diff = 0.0;
//   double max_rel_diff = 0.0;
//
//   for (std::size_t i = 0; i < res_size; ++i) {
//     const double ref  = static_cast<double>(res_ref[i]);
//     const double gpu  = static_cast<double>(res_gpu[i]);
//     const double diff = std::fabs(ref - gpu);
//     max_abs_diff      = std::max(max_abs_diff, diff);
//
//     const double denom = std::max(std::fabs(ref), 1e-20);
//     const double rel   = diff / denom;
//     max_rel_diff       = std::max(max_rel_diff, rel);
//   }
//
//   std::printf("CPU vs GPU comparison:\n");
//   std::printf("  max |diff|   = %.6e\n", max_abs_diff);
//   std::printf("  max rel diff = %.6e\n", max_rel_diff);
//
//   // Optional: assert on a tolerance
//   const double atol = 1e-5;
//   const double rtol = 1e-4;
//
//   if (max_abs_diff > atol && max_rel_diff > rtol) {
//     std::printf("  -> MISMATCH (tolerances: atol=%.1e, rtol=%.1e)\n", atol, rtol);
//   } else {
//     std::printf("  -> OK (within tolerances)\n");
//   }
// }
