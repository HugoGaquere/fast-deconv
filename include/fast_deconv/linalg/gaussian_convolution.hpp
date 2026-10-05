#pragma once
#include <vector>

namespace fast_deconv::linalg {

/// Zero gap, in pixels, beyond which the image-domain kernel of Gaussian(@p sigma) is below float32
/// precision (1e-7 of its peak): a gaussian_convolution_ctx built with gap >= gaussian_reach(sigma) gives
/// the linear convolution for that sigma. 0 for sigma <= 0 (the identity).
int gaussian_reach(float sigma);

/// The gap that makes every Gaussian(@p factor * sigma), sigma in @p sigmas, wrap-free. Not the reach of the
/// largest sigma: below sigma ~1.5 the cut at Nyquist gives a slow tail, and gaussian_reach(0.5) is above
/// gaussian_reach(100).
int max_gaussian_reach(const std::vector<float>& sigmas, float factor = 1.0f);

}  // namespace fast_deconv::linalg

// Resolved by the include path: CMake puts backend/${FAST_DECONV_BACKEND}
// on it, and every backend provides this file.
#include <fd_backend/linalg/gaussian_convolution.hpp>
