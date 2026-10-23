#include <cstddef>
#include <cstdint>
#include <fast_deconv/common/detail/mask_steps.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
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

// slice[i] = !dilated[i] || external[i]: component neighborhoods valid, external pixels masked.
__global__ void finalize_mask_kernel(const bool* dilated, const bool* external, bool* slice, std::int64_t n)
{
  const std::int64_t tid = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  slice[tid] = !dilated[tid] || external[tid];
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::common::detail {

namespace {
constexpr unsigned kBlock = 256;
}  // namespace

void stamp_peaks(const core::exec_ctx& ctx, const std::vector<index2d>& coords, const std::vector<int>& scales,
                 core::span3d<bool> mask_per_scale)
{
  CHECK_CUDA(cudaMemsetAsync(mask_per_scale.data_handle(), 0, mask_per_scale.size() * sizeof(bool), ctx.cuda_stream));
  if (coords.empty()) return;

  // Flat offsets on the host: the kernel then needs neither the coordinates nor the strides.
  std::vector<std::int64_t> h_offsets;
  h_offsets.reserve(coords.size());
  for (std::size_t i = 0; i < coords.size(); i++)
    h_offsets.push_back(mask_per_scale.mapping()(scales.at(i), coords.at(i).row, coords.at(i).col));

  const std::int64_t n_coords = std::ssize(h_offsets);
  auto d_offsets = ctx.alloc_mdcontainer<std::int64_t>(n_coords);
  ctx.copy(d_offsets.data_handle(), h_offsets.data(), n_coords);

  const unsigned grid = core::narrow<unsigned>(core::ceil_div(n_coords, kBlock));
  kernel::build_mask_per_scale_kernel<<<grid, kBlock, 0, ctx.cuda_stream>>>(d_offsets.data_handle(),
                                                                            mask_per_scale.data_handle(), n_coords);
  CHECK_LAST_CUDA_ERROR();
}

void threshold_above(const core::exec_ctx& ctx, core::span2d<const float> in, float threshold, core::span2d<bool> out)
{
  const std::int64_t n = in.mapping().required_span_size();
  const unsigned grid = core::narrow<unsigned>(core::ceil_div(n, kBlock));
  kernel::threshold_above_kernel<<<grid, kBlock, 0, ctx.cuda_stream>>>(in.data_handle(), threshold, out.data_handle(),
                                                                       n);
  CHECK_LAST_CUDA_ERROR();
}

void finalize_mask(const core::exec_ctx& ctx, core::span2d<const bool> dilated, core::span2d<const bool> external,
                   core::span2d<bool> slice)
{
  const std::int64_t n = slice.mapping().required_span_size();
  const unsigned grid = core::narrow<unsigned>(core::ceil_div(n, kBlock));
  kernel::finalize_mask_kernel<<<grid, kBlock, 0, ctx.cuda_stream>>>(dilated.data_handle(), external.data_handle(),
                                                                     slice.data_handle(), n);
  CHECK_LAST_CUDA_ERROR();
}

}  // namespace fast_deconv::common::detail
