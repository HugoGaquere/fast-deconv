#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

// Host reference implementations the unit tests compare GPU results against.
// All accumulate in double so oracle error stays well below test tolerances.

namespace fast_deconv::test {

// Row-major flat index used everywhere: row * ncol + col.
inline int flat(int r, int c, int ncol) { return r * ncol + c; }

// Sum-normalized 2D Gaussian on an (nrow, ncol) grid, centered at (cr, cc).
// sigma == 0 degenerates to a delta at the nearest pixel.
inline std::vector<float> gaussian2d(int nrow, int ncol, double cr, double cc, double sigma)
{
  std::vector<float> img(static_cast<std::size_t>(nrow) * ncol, 0.0f);
  if (sigma == 0.0) {
    img.at(flat(static_cast<int>(std::lround(cr)), static_cast<int>(std::lround(cc)), ncol)) = 1.0f;
    return img;
  }
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

// Zero-padded direct 2D convolution, kernel origin at (kh/2, kw/2) — the same
// center convention as the library's FFT path. Output has the input's shape.
inline std::vector<float> direct_convolve_2d(const std::vector<float>& img, int nrow, int ncol,
                                             const std::vector<float>& kernel, int kh, int kw)
{
  std::vector<float> out(static_cast<std::size_t>(nrow) * ncol, 0.0f);
  for (int r = 0; r < nrow; ++r) {
    for (int c = 0; c < ncol; ++c) {
      double acc = 0.0;
      for (int i = 0; i < kh; ++i) {
        for (int j = 0; j < kw; ++j) {
          const int rr = r - (i - kh / 2);
          const int cc = c - (j - kw / 2);
          if (rr < 0 || rr >= nrow || cc < 0 || cc >= ncol) continue;
          acc += static_cast<double>(kernel.at(flat(i, j, kw))) * img.at(flat(rr, cc, ncol));
        }
      }
      out.at(flat(r, c, ncol)) = static_cast<float>(acc);
    }
  }
  return out;
}

// out[i] = sum_f weights[f] * data[f * npix + i]
inline std::vector<float> weighted_sum(const std::vector<float>& data, const std::vector<float>& weights,
                                       std::size_t npix)
{
  std::vector<float> out(npix, 0.0f);
  for (std::size_t i = 0; i < npix; ++i) {
    double acc = 0.0;
    for (std::size_t f = 0; f < weights.size(); ++f) acc += static_cast<double>(weights.at(f)) * data.at(f * npix + i);
    out.at(i) = static_cast<float>(acc);
  }
  return out;
}

// Max over pixels where mask == 0 (mask nonzero means excluded, matching the
// library convention). Returns -FLT_MAX when every pixel is masked.
inline float masked_max(const std::vector<float>& data, const std::vector<uint8_t>& mask, bool use_abs)
{
  float best = -FLT_MAX;
  for (std::size_t i = 0; i < data.size(); ++i) {
    if (mask.at(i)) continue;
    const float v = use_abs ? std::fabs(data.at(i)) : data.at(i);
    if (v > best) best = v;
  }
  return best;
}

// sqrt(max(E[x^2] - E[x]^2, 0)) over ALL pixels — the matrix::stats_ctx::run
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
