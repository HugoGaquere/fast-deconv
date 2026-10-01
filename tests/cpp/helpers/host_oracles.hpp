#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
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
// library convention). Returns -inf when every pixel is masked.
inline float masked_max(const std::vector<float>& data, const std::vector<uint8_t>& mask, bool use_abs)
{
  float best = -std::numeric_limits<float>::infinity();
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

// Taps h[0..R] of the band-limited Gaussian, the image-domain kernel of exp(-2 pi^2 sigma^2 f^2) for
// f in [-1/2, 1/2] (what the frequency-domain scale convolution applies on an infinite grid):
// h[x] = 2 * integral_0^{1/2} H(f) cos(2 pi f x) df, by composite Simpson. sigma == 0 gives a delta.
inline std::vector<double> band_limited_gaussian_taps(double sigma, int R)
{
  constexpr double pi = 3.14159265358979323846;
  constexpr int M = 4000;  // even; the integrand has at most R / 2 oscillations on [0, 1/2]
  std::vector<double> h(static_cast<std::size_t>(R) + 1);
  for (int x = 0; x <= R; ++x) {
    double acc = 0.0;
    for (int k = 0; k <= M; ++k) {
      const double f = 0.5 * k / M;
      const double w = (k == 0 || k == M) ? 1.0 : (k % 2 ? 4.0 : 2.0);
      acc += w * std::exp(-2.0 * pi * pi * sigma * sigma * f * f) * std::cos(2.0 * pi * f * x);
    }
    h.at(x) = 2.0 * acc * (0.5 / M) / 3.0;
  }
  return h;
}

// Linear (zero outside) separable convolution of each (nrow, ncol) plane of a batch with the symmetric
// taps h[0..R], h[-a] = h[a]: rows then columns, in double.
inline std::vector<float> separable_linear_convolve(const std::vector<float>& img, int batch, int nrow, int ncol,
                                                    const std::vector<double>& h)
{
  const int R = static_cast<int>(h.size()) - 1;
  const auto tap = [&](int d) { return std::abs(d) <= R ? h.at(std::abs(d)) : 0.0; };
  const std::size_t plane = static_cast<std::size_t>(nrow) * ncol;
  std::vector<double> tmp(img.size());
  std::vector<float> out(img.size());
  for (int b = 0; b < batch; ++b) {
    const std::size_t o = b * plane;
    for (int r = 0; r < nrow; ++r)
      for (int c = 0; c < ncol; ++c) {
        double acc = 0.0;
        for (int k = 0; k < ncol; ++k) acc += tap(c - k) * img.at(o + flat(r, k, ncol));
        tmp.at(o + flat(r, c, ncol)) = acc;
      }
    for (int r = 0; r < nrow; ++r)
      for (int c = 0; c < ncol; ++c) {
        double acc = 0.0;
        for (int k = 0; k < nrow; ++k) acc += tap(r - k) * tmp.at(o + flat(k, c, ncol));
        out.at(o + flat(r, c, ncol)) = static_cast<float>(acc);
      }
  }
  return out;
}

}  // namespace fast_deconv::test
