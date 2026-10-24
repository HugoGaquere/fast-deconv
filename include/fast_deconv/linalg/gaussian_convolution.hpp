#pragma once
#include <cstdint>
#include <vector>

namespace fast_deconv::linalg {

/**
 * @brief Zero gap, in pixels, beyond which the image-domain kernel of Gaussian(@p sigma) is below float32
 *        precision (1e-7 of its peak).
 *
 * A zero gap of at least this after the image makes an FFT convolution with that sigma linear.
 *
 * @param[in] sigma  Gaussian sigma in pixels.
 * @return The gap; 0 for sigma <= 0 (the identity).
 */
std::int64_t gaussian_reach(float sigma);

/**
 * @brief The gap that makes every Gaussian(@p factor * sigma), sigma in @p sigmas, wrap-free.
 *
 * Not the reach of the largest sigma: below sigma ~1.5 the cut at Nyquist gives a slow tail, and
 * gaussian_reach(0.5) is above gaussian_reach(100).
 *
 * @param[in] sigmas  Gaussian sigmas in pixels.
 * @param[in] factor  Multiplier applied to every sigma.
 */
std::int64_t max_gaussian_reach(const std::vector<float>& sigmas, float factor = 1.0f);

}  // namespace fast_deconv::linalg

// Resolved by the include path: CMake puts backend/${FAST_DECONV_BACKEND}
// on it, and every backend provides this file.
#include <fd_backend/linalg/gaussian_convolution.hpp>
