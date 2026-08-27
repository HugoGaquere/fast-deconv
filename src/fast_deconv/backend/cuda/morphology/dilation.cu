#include <fast_deconv/morphology/dilation.hpp>
#include <fast_deconv/morphology/roi.hpp>

namespace fast_deconv::kernel {

__global__ void binary_dilation(bool* data, bool* structure, morphology::roi structure_roi, bool* out, int data_stride,
                                int structure_stride, int n)
{
  const int tix = blockIdx.x * blockDim.x + threadIdx.x;
  const int tiy = blockIdx.y * blockDim.y + threadIdx.y;
  const int ncol = data_stride;
  const int nrow = n / data_stride;

  if (tiy >= nrow || tix >= ncol) return;

  const int tid = tiy * data_stride + tix;

  const int height = structure_roi.xmax - structure_roi.xmin;
  const int width = structure_roi.ymax - structure_roi.ymin;

  // fast path, check if data is true
  if (data[tid]) {
    out[tid] = true;
    return;
  }

  const int struct_start_offset = structure_roi.xmin * structure_stride + structure_roi.ymin;
  const int row_start = tiy - height / 2;
  const int col_start = tix - width / 2;

  for (int i = 0; i < height; i++) {
    const int rr = row_start + i;
    if (rr < 0 || rr >= nrow) continue;
    for (int j = 0; j < width; j++) {
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
                     roi structure_roi, core::span2d<bool> out)
{
  const int n = data.extent(0) * data.extent(1);
  dim3 block(32, 8);
  dim3 grid(CEIL_DIV(data.extent(1), block.x), CEIL_DIV(data.extent(0), block.y), 1);
  kernel::binary_dilation<<<grid, block, 0, ctx.cuda_stream>>>(data.data_handle(), structure.data_handle(),
                                                               structure_roi, out.data_handle(), data.extent(1),
                                                               structure.extent(1), n);
}

}  // namespace fast_deconv::morphology