#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/linalg.hpp>

#include "fast_deconv/core/exec_ctx.hpp"

namespace fast_deconv::linalg {

namespace {

// out[i] = sum_c weights[c] * A[c * plane + i] over the first n pixels of each plane.
void weighted_sum_planes(const float* __restrict A, const float* __restrict weights, float* __restrict out,
                         std::int64_t w, std::int64_t plane, std::int64_t n)
{
  // Pixel-major: each thread owns a slice of `out`, and the accumulator stays in a register.
#pragma omp parallel for
  for (std::int64_t i = 0; i < n; i++) {
    float acc = 0.0f;
    for (std::int64_t c = 0; c < w; c++) acc += A[c * plane + i] * weights[c];
    out[i] = acc;
  }
}

}  // namespace

void weighted_sum(const core::exec_ctx& ctx, core::span3d<const float> A, core::span1d<const float> weights,
                  core::span2d<float> out)
{
  FD_PROFILE_FN();
  // TODO: assert that A.extent(0) == weights.size()
  weighted_sum_planes(A.data_handle(), weights.data_handle(), out.data_handle(), weights.extent(0),
                      A.mapping().stride(0), out.mapping().required_span_size());
}

}  // namespace fast_deconv::linalg
