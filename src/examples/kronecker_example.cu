#include "kronecker_example.cuh"

#include <fast_deconv/core/mdarray.hpp>
#include <fast_deconv/core/stream_resources.hpp>
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

template <class T>
inline void kron_cpu(const T* A, const T* B, T* C, int m, int n, int k, int p) noexcept
{
  if (!A || !B || !C) return;
  if (m <= 0 || n <= 0 || k <= 0 || p <= 0) return;

  const std::size_t NP = static_cast<std::size_t>(n) * static_cast<std::size_t>(p);

// Parallelize over (r,c) tiles; collapse helps balance work on big matrices
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      const T a = A[static_cast<std::size_t>(r) * n + c];

      // Precompute bases for this (r,c) tile
      const std::size_t row_block_base = static_cast<std::size_t>(r) * k * NP;  // (r*k)*NP
      const std::size_t col_block_base = static_cast<std::size_t>(c) * p;       // c*p

      // For each u-row of B (and of the k-by-p block)
      for (int u = 0; u < k; ++u) {
        const T* Bb = B + static_cast<std::size_t>(u) * p;

        // Starting linear index in C for row (r*k + u) and col offset (c*p)
        T* Cb = C + (row_block_base + static_cast<std::size_t>(u) * NP + col_block_base);

        // Write a whole p-length row contiguously
        for (int v = 0; v < p; ++v) {
          Cb[v] = a * Bb[v];
        }
      }
    }
  }
}

template <typename T>
bool are_equals(const T* A, const T* B, size_t size)
{
  for (int i = 0; i < size; i++) {
    if (A[i] != B[i]) return false;
  }
  return true;
}

void run_kronecker_example()
{
  std::printf("Running kronecker example\n");
  fast_deconv::core::stream_resources resources;
  uint m     = 10;
  uint n     = 10;
  uint k     = 601;
  uint p     = 601;
  uint total = m * n * p * k;

  auto A     = fast_deconv::core::make_managed_mdarray<float>(m, n);
  auto B     = fast_deconv::core::make_managed_mdarray<float>(k, p);
  auto C_gpu = fast_deconv::core::make_managed_mdarray<float>(m * k, n * p);
  auto C_cpu = fast_deconv::core::make_managed_mdarray<float>(m * k, n * p);

  fill_with_random(A.view());
  fill_with_random(B.view());

  fast_deconv::util::run_benchmark([&]() -> void {
    fast_deconv::linalg::kronecker_async(resources,
                                         A.view().data_handle(),
                                         B.view().data_handle(),
                                         C_gpu.view().data_handle(),
                                         m,
                                         n,
                                         p,
                                         k);
  });

  kron_cpu(A.view().data_handle(), B.view().data_handle(), C_cpu.view().data_handle(), m, n, k, p);

  printf("Equals: %d\n", are_equals(C_gpu.view().data_handle(), C_cpu.view().data_handle(), total));
}
