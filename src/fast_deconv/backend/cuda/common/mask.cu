#include <cstddef>
#include <cstdint>
#include <fast_deconv/common/detail/mask_steps.hpp>
#include <fd_backend/core/launch.cuh>
#include <iterator>
#include <vector>

namespace fast_deconv::kernel {

__global__ void build_mask_per_scale_kernel(const std::int64_t* offsets, bool* out_mask_per_scale,
                                            std::int64_t n_coords)
{
  const std::int64_t tid = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (tid >= n_coords) return;
  out_mask_per_scale[offsets[tid]] = true;
}

// Threshold a real image into a bool mask.
__global__ void threshold_above_kernel(const float* data, float threshold, bool* out, std::int64_t n)
{
  const std::int64_t tid = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  out[tid] = data[tid] > threshold;
}

// out[i] = !region[i] || external[i]: pixels outside the region masked, external pixels masked.
__global__ void mask_outside_kernel(const bool* region, const bool* external, bool* out, std::int64_t n)
{
  const std::int64_t tid = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  out[tid] = !region[tid] || external[tid];
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::common::detail {

void stamp_peaks(const core::exec_ctx& ctx, const std::vector<index2d>& coords, const std::vector<int>& scales,
                 core::span3d<bool> mask_per_scale)
{
  ctx.zero(mask_per_scale);
  if (coords.empty()) return;

  // Flat offsets on the host: the kernel then needs neither the coordinates nor the strides.
  std::vector<std::int64_t> h_offsets;
  h_offsets.reserve(coords.size());
  for (std::size_t i = 0; i < coords.size(); i++)
    h_offsets.push_back(mask_per_scale.mapping()(scales.at(i), coords.at(i).row, coords.at(i).col));

  const std::int64_t n_coords = std::ssize(h_offsets);
  auto d_offsets = ctx.alloc_mdcontainer<std::int64_t>(n_coords);
  ctx.copy(d_offsets.data_handle(), h_offsets.data(), n_coords);

  const core::launch_cfg cfg(n_coords);
  core::launch(ctx, cfg, kernel::build_mask_per_scale_kernel, d_offsets.data_handle(), mask_per_scale.data_handle(),
               n_coords);
}

void threshold_above(const core::exec_ctx& ctx, core::span2d<const float> in, float threshold, core::span2d<bool> out)
{
  const std::int64_t n = in.mapping().required_span_size();
  const core::launch_cfg cfg(n);
  core::launch(ctx, cfg, kernel::threshold_above_kernel, in.data_handle(), threshold, out.data_handle(), n);
}

void mask_outside(const core::exec_ctx& ctx, core::span2d<const bool> region, core::span2d<const bool> external,
                  core::span2d<bool> out)
{
  const std::int64_t n = out.mapping().required_span_size();
  const core::launch_cfg cfg(n);
  core::launch(ctx, cfg, kernel::mask_outside_kernel, region.data_handle(), external.data_handle(), out.data_handle(),
               n);
}

}  // namespace fast_deconv::common::detail
