#pragma once

#include <fast_deconv/matrix/peak.hpp>

// Resolved by the include path: CMake puts backend/${FAST_DECONV_BACKEND}
// on it, and every backend provides this file.
#include <fd_backend/matrix/tiled_argmax.hpp>
