#include <cassert>
#include <cstdint>
#include <fast_deconv/core/macro.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <fd_backend/core/for_each.hpp>

namespace fast_deconv::linalg {

void weighted_sum(const core::exec_ctx& ctx, core::span3d<const float> A, core::span1d<const float> weights,
                  core::span2d<float> out)
{
  FD_PROFILE_FN();
  assert(A.extent(0) == weights.extent(0));
  assert(out.is_exhaustive());
  const std::int64_t n_planes = weights.extent(0);
  const std::int64_t plane = A.mapping().stride(0);
  const float* a = A.data_handle();
  const float* w = weights.data_handle();
  float* o = out.data_handle();

  // Pixel-major: each pixel's accumulator stays in a register, and on CUDA a warp reads each plane contiguously.
  core::for_each_index(ctx, out.mapping().required_span_size(), [=] FD_HOST_DEVICE(std::int64_t i) {
    float acc = 0.0f;
    for (std::int64_t c = 0; c < n_planes; c++) acc += a[c * plane + i] * w[c];
    o[i] = acc;
  });
}

}  // namespace fast_deconv::linalg
