#pragma once

#include <cub/cub.cuh>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::matrix {

/// Packed reduction state combining the four contributions needed to compute
/// max + rms in a single CUB sweep.
struct stats_acc {
  float max_v;
  float sum;
  float sum_sq;
  int count;
};

struct stats_combine {
  __host__ __device__ __forceinline__ stats_acc operator()(const stats_acc& a, const stats_acc& b) const
  {
    return {fmaxf(a.max_v, b.max_v), a.sum + b.sum, a.sum_sq + b.sum_sq, a.count + b.count};
  }
};

/// Result returned to the host after a fused max + rms reduction.
struct stats_result {
  float max;  ///< masked max (with optional abs); -FLT_MAX if every pixel is masked
  float rms;  ///< sqrt(max(E[x^2] - E[x]^2, 0)) over all pixels (mask not applied)
};

/// Pre-allocated workspace for the fused max + rms reduction over a 2D image
/// with a boolean mask. Mirrors the shape of @ref argmax_workspace: temp
/// storage and the packed-state output are allocated once at construction and
/// reused across calls.
///
/// `use_abs` is fixed at construction so the iterator type stays stable and
/// the CUB temp-bytes query computed in the constructor stays valid for every
/// subsequent call.
struct stats_workspace {
  const core::stream_resources& stream_res;
  size_t n_elements = 0;
  bool use_abs = false;

  core::device_cont<stats_acc> d_state;
  core::device_cont<char> d_temp;
  size_t temp_storage_bytes = 0;

  stats_acc h_state{};

  stats_workspace(const core::stream_resources& stream_res, size_t n_elements, bool use_abs);

  stats_workspace(const stats_workspace&) = delete;
  stats_workspace& operator=(const stats_workspace&) = delete;
};

/// Issue the fused reduction onto `ws.stream_res`. No host sync; the result
/// lands in `ws.d_state` on device.
void compute_stats_async(stats_workspace& ws, core::device_span2d<float> data, core::device_span2d<bool> mask);

/// Async issue + D2H copy + stream sync. Returns {max, rms} on host.
stats_result compute_stats(stats_workspace& ws, core::device_span2d<float> data, core::device_span2d<bool> mask);

}  // namespace fast_deconv::matrix
