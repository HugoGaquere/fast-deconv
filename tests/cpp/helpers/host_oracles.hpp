#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

// Host reference implementations shared by several tests. They accumulate in double, so their own
// error stays well below the test tolerances.

namespace fast_deconv::test {

// Row-major flat index used everywhere: row * ncol + col.
inline int flat(int r, int c, int ncol) { return r * ncol + c; }

// Sum-normalized 2D Gaussian on an (nrow, ncol) grid, centered at (cr, cc).
inline std::vector<float> gaussian2d(int nrow, int ncol, double cr, double cc, double sigma)
{
  std::vector<float> img(static_cast<std::size_t>(nrow) * ncol);
  std::vector<double> tmp(img.size());
  double sum = 0.0;
  for (int r = 0; r < nrow; ++r) {
    for (int c = 0; c < ncol; ++c) {
      const double d2 = (r - cr) * (r - cr) + (c - cc) * (c - cc);
      const double v = std::exp(-d2 / (2.0 * sigma * sigma));
      tmp.at(flat(r, c, ncol)) = v;
      sum += v;
    }
  }
  for (std::size_t i = 0; i < img.size(); ++i) img.at(i) = static_cast<float>(tmp.at(i) / sum);
  return img;
}

// sqrt(max(E[x^2] - E[x]^2, 0)) over ALL pixels — the matrix::stats_ctx::max_and_rms
// rms definition (its mask applies to the max only).
inline float std_all(const std::vector<float>& data)
{
  double sum = 0.0, sum_sq = 0.0;
  for (const float v : data) {
    sum += v;
    sum_sq += static_cast<double>(v) * v;
  }
  const double n = static_cast<double>(data.size());
  const double var = sum_sq / n - (sum / n) * (sum / n);
  return static_cast<float>(std::sqrt(var > 0.0 ? var : 0.0));
}

}  // namespace fast_deconv::test
