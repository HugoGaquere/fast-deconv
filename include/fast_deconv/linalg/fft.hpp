#pragma once
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>

// Resolved by the include path: CMake puts backend/${FAST_DECONV_BACKEND}
// on it, and every backend provides this file.
#include <fd_backend/linalg/fft.hpp>
