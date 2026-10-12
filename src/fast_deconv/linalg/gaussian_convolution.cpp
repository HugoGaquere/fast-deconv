#include <algorithm>
#include <cmath>
#include <fast_deconv/linalg/gaussian_convolution.hpp>
#include <numbers>

namespace fast_deconv::linalg {

// Last pixel where the image-domain kernel of exp(-2 pi^2 sigma^2 f^2), f in [-1/2, 1/2], is above
// kReachTol of its peak: float32 precision. Closed form, an upper bound on the measured reach (within
// 1-3 px for sigma in [0.05, 200], see gaussian_conv/padding_and_kernel_reach.ipynb):
//   - the Gaussian decay: exp(-x^2 / 2 sigma^2) = tol  =>  x = sigma sqrt(2 ln(1/tol));
//   - the cut at Nyquist (small sigma): |h(x)| ~ sigma^2 H(1/2) / x^2, against the peak
//     h(0) = erf(pi sigma / sqrt 2) / (sigma sqrt(2 pi)).
int gaussian_reach(float sigma)
{
  constexpr double kReachTol = 1e-7;
  if (!(sigma > 0.0f)) return 0;  // Gaussian(0) is the identity
  const double s = sigma;
  const double pi = std::numbers::pi;
  const double gaussian = s * std::sqrt(2.0 * std::log(1.0 / kReachTol));
  const double peak = std::erf(pi * s / std::sqrt(2.0)) / (s * std::sqrt(2.0 * pi));
  const double h_nyquist = std::exp(-pi * pi * s * s / 2.0);
  const double tail = std::sqrt(s * s * h_nyquist / (kReachTol * peak));
  return static_cast<int>(std::ceil(std::max(gaussian, tail)));
}

int max_gaussian_reach(const std::vector<float>& sigmas, float factor)
{
  int reach = 0;
  for (float sigma : sigmas) reach = std::max(reach, gaussian_reach(factor * sigma));
  return reach;
}

}  // namespace fast_deconv::linalg
