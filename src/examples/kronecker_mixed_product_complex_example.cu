// #include "kronecker_mixed_product_example.cuh"
//
// #include <cuComplex.h>
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
// #include <cstdlib>
// #include <limits>
// #include <vector>
//
// // Row-major index helper
// inline std::size_t idx(std::size_t i, std::size_t j, std::size_t ncols) { return i * ncols + j; }
//
// // Fill managed mdarray (or view) with random complex values
// template <typename View>
// void fill_with_random(View arr)
// {
//   for (int i = 0; i < arr.size(); i++) {
//     float re             = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
//     float im             = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
//     arr.data_handle()[i] = make_cuFloatComplex(re, im);
//   }
// }
//
// // Plain CPU GEMM (row-major) for cuFloatComplex:
// // C[m,n] = A[m,k] * B[k,n]
// void gemm_row_major_cpu(const cuFloatComplex* A,
//                         const cuFloatComplex* B,
//                         cuFloatComplex* C,
//                         std::size_t m,
//                         std::size_t n,
//                         std::size_t k)
// {
//   for (std::size_t i = 0; i < m; ++i) {
//     for (std::size_t j = 0; j < n; ++j) {
//       cuFloatComplex acc = make_cuFloatComplex(0.0f, 0.0f);
//       for (std::size_t kk = 0; kk < k; ++kk) {
//         cuFloatComplex a = A[idx(i, kk, k)];
//         cuFloatComplex b = B[idx(kk, j, n)];
//         acc              = cuCaddf(acc, cuCmulf(a, b));
//       }
//       C[idx(i, j, n)] = acc;
//     }
//   }
// }
//
// // CPU Kronecker for mixed-product (complex):
// // res[(a*b) x (c*d)] = kron(AC[a,c], BD[b,d])
// void kronecker_mixed_product_cpu(const cuFloatComplex* AC,
//                                  const cuFloatComplex* BD,
//                                  cuFloatComplex* res,
//                                  std::size_t a_rows,
//                                  std::size_t c_cols,
//                                  std::size_t b_rows,
//                                  std::size_t d_cols)
// {
//   const std::size_t res_rows = a_rows * b_rows;
//   const std::size_t res_cols = c_cols * d_cols;
//
//   (void)res_rows;  // unused but kept for clarity
//
//   for (std::size_t iA = 0; iA < a_rows; ++iA) {
//     for (std::size_t jA = 0; jA < c_cols; ++jA) {
//       const cuFloatComplex ac_val = AC[idx(iA, jA, c_cols)];
//
//       for (std::size_t iB = 0; iB < b_rows; ++iB) {
//         for (std::size_t jB = 0; jB < d_cols; ++jB) {
//           const cuFloatComplex bd_val = BD[idx(iB, jB, d_cols)];
//
//           const std::size_t i_res = iA * b_rows + iB;
//           const std::size_t j_res = jA * d_cols + jB;
//
//           res[idx(i_res, j_res, res_cols)] = cuCmulf(ac_val, bd_val);
//         }
//       }
//     }
//   }
// }
//
// void run_kronecker_mixed_product_complex_example()
// {
//   std::printf("Running kronecker Mixed Product example (cuFloatComplex)\n");
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
//   auto A = fast_deconv::core::make_managed_mdarray<cuFloatComplex>(a_rows, a_cols);
//   auto B = fast_deconv::core::make_managed_mdarray<cuFloatComplex>(b_rows, b_cols);
//   auto C = fast_deconv::core::make_managed_mdarray<cuFloatComplex>(c_rows, c_cols);
//   auto D = fast_deconv::core::make_managed_mdarray<cuFloatComplex>(d_rows, d_cols);
//
//   auto AC = fast_deconv::core::make_managed_mdarray<cuFloatComplex>(a_rows, c_cols);
//   auto BD = fast_deconv::core::make_managed_mdarray<cuFloatComplex>(b_rows, d_cols);
//
//   auto res =
//     fast_deconv::core::make_managed_mdarray<cuFloatComplex>(a_rows * b_rows, c_cols * d_cols);
//
//   fill_with_random(A.view());
//   fill_with_random(B.view());
//   fill_with_random(C.view());
//   fill_with_random(D.view());
//
//   fast_deconv::core::stream_resources resources;
//
//   fast_deconv::util::run_benchmark(
//     [&]() -> void {
//       fast_deconv::linalg::cgemm_row_major_async(resources,
//                                                  a_rows,
//                                                  c_cols,
//                                                  a_cols,
//                                                  A.view().data_handle(),
//                                                  C.view().data_handle(),
//                                                  AC.view().data_handle());
//
//       fast_deconv::linalg::cgemm_row_major_async(resources,
//                                                  b_rows,
//                                                  d_cols,
//                                                  b_cols,
//                                                  B.view().data_handle(),
//                                                  D.view().data_handle(),
//                                                  BD.view().data_handle());
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
//   resources.sync();
//
//   // ---- CPU reference ----
//
//   std::vector<cuFloatComplex> AC_ref(a_rows * c_cols);
//   std::vector<cuFloatComplex> BD_ref(b_rows * d_cols);
//   std::vector<cuFloatComplex> res_ref(a_rows * b_rows * c_cols * d_cols);
//
//   cuFloatComplex* A_ptr   = A.view().data_handle();
//   cuFloatComplex* B_ptr   = B.view().data_handle();
//   cuFloatComplex* C_ptr   = C.view().data_handle();
//   cuFloatComplex* D_ptr   = D.view().data_handle();
//   cuFloatComplex* res_gpu = res.view().data_handle();
//
//   gemm_row_major_cpu(A_ptr, C_ptr, AC_ref.data(), a_rows, c_cols, a_cols);
//   gemm_row_major_cpu(B_ptr, D_ptr, BD_ref.data(), b_rows, d_cols, b_cols);
//   kronecker_mixed_product_cpu(
//     AC_ref.data(), BD_ref.data(), res_ref.data(), a_rows, c_cols, b_rows, d_cols);
//
//   // ---- Compare GPU result vs CPU reference ----
//   const uint res_size = a_rows * b_rows * c_cols * d_cols;
//
//   double max_abs_diff = 0.0;
//   double max_rel_diff = 0.0;
//
//   for (uint i = 0; i < res_size; ++i) {
//     cuFloatComplex ref_c = res_ref[i];
//     cuFloatComplex gpu_c = res_gpu[i];
//
//     cuFloatComplex diff_c = cuCsubf(ref_c, gpu_c);
//     auto diff_mag         = static_cast<double>(cuCabsf(diff_c));
//     max_abs_diff          = std::max(max_abs_diff, diff_mag);
//
//     auto ref_mag = static_cast<double>(cuCabsf(ref_c));
//     double denom = std::max(ref_mag, 1e-20);
//     double rel   = diff_mag / denom;
//     max_rel_diff = std::max(max_rel_diff, rel);
//   }
//
//   std::printf("CPU vs GPU comparison (complex):\n");
//   std::printf("  max |diff|   = %.6e\n", max_abs_diff);
//   std::printf("  max rel diff = %.6e\n", max_rel_diff);
//
//   const double atol = 1e-5;
//   const double rtol = 1e-4;
//
//   if (max_abs_diff > atol && max_rel_diff > rtol) {
//     std::printf("  -> MISMATCH (tolerances: atol=%.1e, rtol=%.1e)\n", atol, rtol);
//   } else {
//     std::printf("  -> OK (within tolerances)\n");
//   }
// }
