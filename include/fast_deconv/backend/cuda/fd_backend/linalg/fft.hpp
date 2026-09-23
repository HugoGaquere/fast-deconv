#pragma once
#include <cufft.h>

#include <cstddef>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>

namespace fast_deconv::linalg {

using complex_type = cufftComplex;

}  // namespace fast_deconv::linalg
