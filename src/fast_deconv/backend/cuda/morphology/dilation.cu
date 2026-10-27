#include <cstdint>
#include <fast_deconv/morphology/dilation.hpp>
#include <fast_deconv/morphology/roi.hpp>
#include <fd_backend/core/launch.cuh>

namespace fast_deconv::kernel {

__global__ void binary_dilation(const bool* data, const bool* structure, common::roi structure_roi, bool* out,
                                std::int64_t nrow, std::int64_t ncol, std::int64_t data_stride,
                                std::int64_t structure_stride)
{
  const std::int64_t tix = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  const std::int64_t tiy = std::int64_t{blockIdx.y} * blockDim.y + threadIdx.y;

  if (tiy >= nrow || tix >= ncol) return;

  const std::int64_t tid = tiy * data_stride + tix;

  const std::int64_t nrow_se = structure_roi.rmax - structure_roi.rmin;
  const std::int64_t ncol_se = structure_roi.cmax - structure_roi.cmin;

  if (data[tid]) {
    out[tid] = true;
    return;
  }

  const std::int64_t struct_start_offset = structure_roi.rmin * structure_stride + structure_roi.cmin;
  const std::int64_t row_start = tiy - nrow_se / 2;
  const std::int64_t col_start = tix - ncol_se / 2;

  for (std::int64_t i = 0; i < nrow_se; i++) {
    const std::int64_t rr = row_start + i;
    if (rr < 0 || rr >= nrow) continue;
    for (std::int64_t j = 0; j < ncol_se; j++) {
      const std::int64_t cc = col_start + j;
      if (cc < 0 || cc >= ncol) continue;
      const std::int64_t structure_idx = struct_start_offset + i * structure_stride + j;
      const std::int64_t data_idx = rr * data_stride + cc;
      if (data[data_idx] && structure[structure_idx]) {
        out[tid] = true;
        return;
      }
    }
  }
}
}  // namespace fast_deconv::kernel

namespace fast_deconv::morphology {

void binary_dilation(const core::exec_ctx& ctx, core::span2d<const bool> data, core::span2d<const bool> structure,
                     common::roi structure_roi, core::span2d<bool> out)
{
  // The kernel only sets pixels, so clear the output first.
  ctx.zero(out);
  const core::launch_cfg cfg(data.extent(0), data.extent(1));
  core::launch(ctx, cfg, kernel::binary_dilation, data.data_handle(), structure.data_handle(), structure_roi,
               out.data_handle(), data.extent(0), data.extent(1), data.stride(0), structure.stride(0));
}

}  // namespace fast_deconv::morphology