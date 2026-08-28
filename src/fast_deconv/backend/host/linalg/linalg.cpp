#include <algorithm>
#include <cstdint>
#include <fast_deconv/linalg/linalg.hpp>

namespace fast_deconv::linalg {

void weighted_sum_async(const core::exec_ctx& ctx, const float* __restrict A, const float* __restrict weights,
                        float* __restrict out, int w, int n)
{
  std::fill_n(out, n, 0.0f);

  for (int c = 0; c < w; c++) {
    const float wc = weights[c];
    for (int i = 0; i < n; i++) {
      out[i] += A[c * n + i] * wc;
    }
  }
}

void weighted_sum_async(const core::exec_ctx& ctx, const core::span3d<const float> A,
                        const core::span1d<const float> weights, core::span2d<float> out)
{
  weighted_sum_async(ctx, A.data_handle(), weights.data_handle(), out.data_handle(), static_cast<int>(weights.size()),
                     static_cast<int>(static_cast<std::int64_t>(A.extent(1)) * A.extent(2)));
}

}  // namespace fast_deconv::linalg
