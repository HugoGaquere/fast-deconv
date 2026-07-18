#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <fast_deconv/linalg/pseudo_inverse.hpp>
#include <random>
#include <vector>

#include "helpers/device_buffers.hpp"
#include "helpers/gpu_test.hpp"
#include "helpers/rng.hpp"

namespace linalg = fast_deconv::linalg;
namespace fdtest = fast_deconv::test;

namespace {

// Host double-precision pseudo-inverse for the shapes used here (the small
// Gram matrix is at most 2x2). Returns P = pinv(A) as row-major [n_cols, n_rows].
//   overdetermined  (n_rows >= n_cols): P = inv(A^T A) A^T, Gram is [n_cols, n_cols]
//   underdetermined (n_rows <  n_cols): P = A^T inv(A A^T), Gram is [n_rows, n_rows]
std::vector<double> host_pinv_2(const std::vector<double>& a, int n_rows, int n_cols)
{
  const bool over = n_rows >= n_cols;
  const int k = over ? n_cols : n_rows;  // Gram size
  EXPECT_EQ(k, 2) << "host oracle only supports a 2x2 Gram matrix";

  // Gram: G = A^T A (over) or A A^T (under).
  double g[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int m = 0; m < (over ? n_rows : n_cols); ++m)
        g[i][j] += over ? a.at(m * n_cols + i) * a.at(m * n_cols + j) : a.at(i * n_cols + m) * a.at(j * n_cols + m);

  const double det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
  const double gi[2][2] = {{g[1][1] / det, -g[0][1] / det}, {-g[1][0] / det, g[0][0] / det}};

  std::vector<double> p(static_cast<std::size_t>(n_cols) * n_rows, 0.0);
  for (int o = 0; o < n_cols; ++o) {
    for (int f = 0; f < n_rows; ++f) {
      double acc = 0.0;
      if (over) {
        for (int j = 0; j < 2; ++j) acc += gi[o][j] * a.at(f * n_cols + j);  // inv(G) A^T
      } else {
        for (int j = 0; j < 2; ++j) acc += a.at(j * n_cols + o) * gi[j][f];  // A^T inv(G)
      }
      p.at(o * n_rows + f) = acc;
    }
  }
  return p;
}

}  // namespace

class PseudoInverse : public fdtest::GpuTest {
 protected:
  // Run compute_pseudo_inverse on row-major A [n_rows, n_cols]; return P as
  // row-major [n_cols, n_rows] (the device output is col-major [n_cols, n_rows],
  // i.e. element (o, f) lives at [f * n_cols + o]).
  std::vector<float> run(const std::vector<float>& a, int n_rows, int n_cols)
  {
    const auto sr = res().make_stream();
    fdtest::device_buffer<float> d_a(res(), sr, a);
    fdtest::device_buffer<float> d_pinv(res(), sr, static_cast<std::size_t>(n_rows) * n_cols);

    linalg::compute_pseudo_inverse(sr, d_a.get(), d_pinv.get(), n_rows, n_cols);
    sr.sync();

    const auto colmajor = d_pinv.to_host();
    std::vector<float> p(colmajor.size());
    for (int o = 0; o < n_cols; ++o)
      for (int f = 0; f < n_rows; ++f) p.at(o * n_rows + f) = colmajor.at(f * n_cols + o);
    return p;
  }
};

TEST_F(PseudoInverse, SquareMatrixGivesInverse)
{
  // A = [[3, 1], [2, 4]], det = 10 → inv = [[0.4, -0.1], [-0.2, 0.3]].
  const std::vector<float> a = {3.0f, 1.0f, 2.0f, 4.0f};
  const auto p = run(a, 2, 2);
  EXPECT_NEAR(p.at(0), 0.4f, 1e-4f);
  EXPECT_NEAR(p.at(1), -0.1f, 1e-4f);
  EXPECT_NEAR(p.at(2), -0.2f, 1e-4f);
  EXPECT_NEAR(p.at(3), 0.3f, 1e-4f);
}

TEST_F(PseudoInverse, OverdeterminedMatchesNormalEquationsOracle)
{
  // Vandermonde-like design matrix (columns {1, log-frequency}), the same shape
  // fit_coefficients feeds through this path.
  const int n_rows = 4, n_cols = 2;
  const std::vector<double> a_d = {1.0, 0.0, 1.0, 0.1, 1.0, 0.2, 1.0, 0.3};
  const std::vector<float> a(a_d.begin(), a_d.end());

  const auto p = run(a, n_rows, n_cols);
  const auto expected = host_pinv_2(a_d, n_rows, n_cols);
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(p.at(i), static_cast<float>(expected.at(i)), 1e-4f) << "element " << i;
}

TEST_F(PseudoInverse, UnderdeterminedTakesTransposeGramPath)
{
  // n_rows < n_cols exercises the A^T inv(A A^T) branch.
  const int n_rows = 2, n_cols = 4;
  const std::vector<double> a_d = {1.0, 0.5, -0.3, 0.8, 0.2, -1.0, 0.7, 0.4};
  const std::vector<float> a(a_d.begin(), a_d.end());

  const auto p = run(a, n_rows, n_cols);
  const auto expected = host_pinv_2(a_d, n_rows, n_cols);
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(p.at(i), static_cast<float>(expected.at(i)), 1e-4f) << "element " << i;
}

// Moore–Penrose identities checked directly (no host pinv needed), robust to
// any internal ordering choice: A P A ≈ A and P A P ≈ P.
TEST_F(PseudoInverse, MoorePenroseIdentitiesOnRandomMatrix)
{
  const int n_rows = 5, n_cols = 3;
  std::mt19937 rng(42);
  std::vector<float> a(n_rows * n_cols);
  fdtest::fill_uniform(rng, a, -1.0f, 1.0f);

  const auto p = run(a, n_rows, n_cols);  // row-major [n_cols, n_rows]

  // apa = A @ P @ A, computed in double.
  auto matmul = [](const std::vector<float>& x, int xr, int xc, const std::vector<float>& y, int yc) {
    std::vector<double> out(static_cast<std::size_t>(xr) * yc, 0.0);
    for (int i = 0; i < xr; ++i)
      for (int j = 0; j < yc; ++j) {
        double acc = 0.0;
        for (int m = 0; m < xc; ++m) acc += static_cast<double>(x.at(i * xc + m)) * y.at(m * yc + j);
        out.at(i * yc + j) = acc;
      }
    return out;
  };

  const auto pa = matmul(p, n_cols, n_rows, a, n_cols);  // [n_cols, n_cols]
  const std::vector<float> pa_f(pa.begin(), pa.end());
  const auto apa = matmul(a, n_rows, n_cols, pa_f, n_cols);  // [n_rows, n_cols]
  for (int i = 0; i < n_rows * n_cols; ++i) EXPECT_NEAR(apa.at(i), a.at(i), 1e-3) << "A·P·A element " << i;

  const auto ap = matmul(a, n_rows, n_cols, p, n_rows);  // [n_rows, n_rows]
  const std::vector<float> ap_f(ap.begin(), ap.end());
  const auto pap = matmul(p, n_cols, n_rows, ap_f, n_rows);  // [n_cols, n_rows]
  for (int i = 0; i < n_cols * n_rows; ++i) EXPECT_NEAR(pap.at(i), p.at(i), 1e-3) << "P·A·P element " << i;
}
