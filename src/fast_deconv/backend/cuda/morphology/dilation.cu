#include <fast_deconv/morphology/dilation.hpp>
#include <fast_deconv/morphology/roi.hpp>

namespace fast_deconv::kernel {

__global__ void binary_dilation(bool* data, bool* structure, common::roi structure_roi, bool* out, int data_stride,
                                int structure_stride, int n)
{
  const int tix = blockIdx.x * blockDim.x + threadIdx.x;
  const int tiy = blockIdx.y * blockDim.y + threadIdx.y;
  const int ncol = data_stride;
  const int nrow = n / data_stride;

  if (tiy >= nrow || tix >= ncol) return;

  const int tid = tiy * data_stride + tix;

  const int nrow_se = structure_roi.rmax - structure_roi.rmin;
  const int ncol_se = structure_roi.cmax - structure_roi.cmin;

  // fast path, check if data is true
  if (data[tid]) {
    out[tid] = true;
    return;
  }

  const int struct_start_offset = structure_roi.rmin * structure_stride + structure_roi.cmin;
  const int row_start = tiy - nrow_se / 2;
  const int col_start = tix - ncol_se / 2;

  for (int i = 0; i < nrow_se; i++) {
    const int rr = row_start + i;
    if (rr < 0 || rr >= nrow) continue;
    for (int j = 0; j < ncol_se; j++) {
      const int cc = col_start + j;
      if (cc < 0 || cc >= ncol) continue;
      const int structure_idx = struct_start_offset + i * structure_stride + j;
      const int data_idx = rr * data_stride + cc;
      if (data[data_idx] && structure[structure_idx]) {
        out[tid] = true;
        return;
      }
    }
  }
}
}  // namespace fast_deconv::kernel

namespace fast_deconv::morphology {

void binary_dilation(const core::exec_ctx& ctx, core::span2d<bool> data, core::span2d<bool> structure,
                     common::roi structure_roi, core::span2d<bool> out)
{
  const int n = data.extent(0) * data.extent(1);
  dim3 block(32, 8);
  dim3 grid(CEIL_DIV(data.extent(1), block.x), CEIL_DIV(data.extent(0), block.y), 1);
  kernel::binary_dilation<<<grid, block, 0, ctx.cuda_stream>>>(data.data_handle(), structure.data_handle(),
                                                               structure_roi, out.data_handle(), data.extent(1),
                                                               structure.extent(1), n);
}

}  // namespace fast_deconv::morphology