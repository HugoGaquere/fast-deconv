#include <gtest/gtest.h>

#include <fast_deconv/linalg/pseudo_inverse.hpp>
#include <random>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/device_buffers.hpp"
#include "helpers/rng.hpp"

namespace linalg = fast_deconv::linalg;
namespace fdtest = fast_deconv::test;

class PseudoInverse : public fdtest::BackendTest {
 protected:
  // Run compute_pseudo_inverse on row-major A [n_rows, n_cols]; return P as
  // row-major [n_cols, n_rows] (the device output is col-major [n_cols, n_rows],
  // i.e. element (o, f) lives at [f * n_cols + o]).
  std::vector<float> run(const std::vector<float>& a, int n_rows, int n_cols)
  {
    const auto sr = res().make_ctx();
    fdtest::device_buffer<float> d_a(sr, a);
    fdtest::device_buffer<float> d_pinv(sr, static_cast<std::size_t>(n_rows) * n_cols);

    linalg::compute_pseudo_inverse(sr, d_a.get(), d_pinv.get(), n_rows, n_cols);
    sr.wait();

    const auto colmajor = d_pinv.to_host();
    std::vector<float> p(colmajor.size());
    for (int o = 0; o < n_cols; ++o)
      for (int f = 0; f < n_rows; ++f) p.at(o * n_rows + f) = colmajor.at(f * n_cols + o);
    return p;
  }

  // Moore–Penrose identities A P A ≈ A and P A P ≈ P, checked in double. They
  // pin the pseudo-inverse for any shape without a host oracle, and are robust
  // to any internal ordering choice.
  void expect_moore_penrose(const std::vector<float>& a, int n_rows, int n_cols)
  {
    const auto p = run(a, n_rows, n_cols);  // row-major [n_cols, n_rows]

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

// Vandermonde-like design matrix (columns {1, log-frequency}). fit_coefficients no
// longer routes through here — it solves in-kernel — so this only covers the
// standalone routine's inv(A^T A) A^T branch.
TEST_F(PseudoInverse, OverdeterminedSatisfiesMoorePenrose)
{
  expect_moore_penrose({1.0f, 0.0f, 1.0f, 0.1f, 1.0f, 0.2f, 1.0f, 0.3f}, 4, 2);
}

// n_rows < n_cols exercises the A^T inv(A A^T) branch.
TEST_F(PseudoInverse, UnderdeterminedTakesTransposeGramPath)
{
  expect_moore_penrose({1.0f, 0.5f, -0.3f, 0.8f, 0.2f, -1.0f, 0.7f, 0.4f}, 2, 4);
}

TEST_F(PseudoInverse, RandomOverdeterminedMatrix)
{
  std::mt19937 rng(42);
  std::vector<float> a(5 * 3);
  fdtest::fill_uniform(rng, a, -1.0f, 1.0f);
  expect_moore_penrose(a, 5, 3);
}
