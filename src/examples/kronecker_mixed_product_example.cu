#include "kronecker_mixed_product_example.cuh"

#include <fast_deconv/core/mdarray.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/linalg/gemm.hpp>
#include <fast_deconv/linalg/kronecker.cuh>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fast_deconv/util/kernel_bench.hpp>

#include <cstdio>

template <typename T>
void fill_with_random(T arr)
{
  for (int i = 0; i < arr.size(); i++) {
    arr.data_handle()[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
}

void run_kronecker_mixed_product_example()
{
  std::printf("Running kronecker Mixed Product example\n");
  fast_deconv::core::stream_resources resources;

  uint a_rows = 10;
  uint a_cols = 10;
  uint b_rows = 601;
  uint b_cols = 601;
  uint c_rows = 10;
  uint c_cols = 20;
  uint d_rows = 601;
  uint d_cols = 61;

  auto A = fast_deconv::core::make_managed_mdarray<float>(a_rows, a_cols);
  auto B = fast_deconv::core::make_managed_mdarray<float>(b_rows, b_cols);
  auto C = fast_deconv::core::make_managed_mdarray<float>(c_rows, c_cols);
  auto D = fast_deconv::core::make_managed_mdarray<float>(d_rows, d_cols);

  auto AC = fast_deconv::core::make_managed_mdarray<float>(a_rows, c_cols);
  auto BD = fast_deconv::core::make_managed_mdarray<float>(b_rows, d_cols);

  auto res = fast_deconv::core::make_managed_mdarray<float>(a_rows * b_rows, c_cols * d_cols);

  fill_with_random(A.view());
  fill_with_random(B.view());
  fill_with_random(C.view());
  fill_with_random(D.view());

  fast_deconv::util::run_benchmark(
    [&]() -> void {
      fast_deconv::linalg::gemm_row_major_async(resources,
                                                a_rows,
                                                c_cols,
                                                a_cols,
                                                A.view().data_handle(),
                                                C.view().data_handle(),
                                                AC.view().data_handle());

      fast_deconv::linalg::gemm_row_major_async(resources,
                                                b_rows,
                                                d_cols,
                                                b_cols,
                                                B.view().data_handle(),
                                                D.view().data_handle(),
                                                BD.view().data_handle());

      fast_deconv::linalg::kronecker_async(resources,
                                           AC.view().data_handle(),
                                           BD.view().data_handle(),
                                           res.view().data_handle(),
                                           a_rows,
                                           c_cols,
                                           b_rows,
                                           d_cols);
    },
    resources.stream);
}
