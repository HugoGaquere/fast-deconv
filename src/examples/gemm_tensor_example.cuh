#pragma once

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
void run_kronecker_tensor();
