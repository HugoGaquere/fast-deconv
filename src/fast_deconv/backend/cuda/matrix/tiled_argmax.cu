#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>
#include <fast_deconv/matrix/tiled_argmax.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace {
// Threads per block for the per-tile reduction. A block reduces its whole tile
// with a grid-stride loop, so this is INDEPENDENT of the tile size: pick it for
// occupancy, not to "cover" the tile. 256 is a safe default.
constexpr int kThreadsPerBlock = 256;
}  // namespace

namespace fast_deconv::kernel {

using fast_deconv::matrix::peak;

// One block per tile. The grid covers the tile rectangle to (re)compute, origin (tile_x0, tile_y0): the whole
// grid for run(), the dirty sub-grid for run_incremental(). Each block writes its tile's peak, index into the
// full image, to d_tiles[tile_y * n_tiles_x + tile_x].
__global__ void tiled_argmax_reduce(const float* data, matrix::peak_criterion criterion, peak* d_tiles,
                                    std::int64_t image_width, std::int64_t image_height, std::int64_t tile_width,
                                    std::int64_t tile_height, std::int64_t n_tiles_x, std::int64_t tile_x0,
                                    std::int64_t tile_y0)
{
  const std::int64_t tile_x = tile_x0 + blockIdx.x;
  const std::int64_t tile_y = tile_y0 + blockIdx.y;
  const std::int64_t ox = tile_x * tile_width;
  const std::int64_t oy = tile_y * tile_height;
  const std::int64_t ex = min(ox + tile_width, image_width);
  const std::int64_t ey = min(oy + tile_height, image_height);
  const std::int64_t tw = ex - ox, th = ey - oy;

  const std::int64_t step = blockDim.x;
  const std::int64_t dcol = step % tw;
  const std::int64_t drow = step / tw;
  std::int64_t col = threadIdx.x % tw;
  std::int64_t row = threadIdx.x / tw;

  peak best{.index = 0, .value = -INFINITY};
  while (row < th) {
    const std::int64_t g = (oy + row) * image_width + (ox + col);
    const float x = data[g];
    const float v = criterion(x, g);
    if (v > best.value) best = {.index = g, .value = v, .signed_value = x};
    row += drow;
    col += dcol;
    if (col >= tw) {
      col -= tw;
      row++;
    }
  }

  using BlockReduce = cub::BlockReduce<peak, kThreadsPerBlock>;
  __shared__ typename BlockReduce::TempStorage temp;
  peak winner = BlockReduce(temp).Reduce(best, matrix::peak_max{});

  if (threadIdx.x == 0) d_tiles[tile_y * n_tiles_x + tile_x] = winner;
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::matrix {

tiled_argmax_impl::tiled_argmax_impl(const core::exec_ctx& ctx, const tile_grid& grid)
    : ctx_(ctx), grid_(grid), n_tiles_(grid.n_tiles_x * grid.n_tiles_y)
{
  d_tiles_ = ctx_.alloc_ptr<peak>(n_tiles_);
  d_result_ = ctx_.alloc_ptr<peak>(1);

  // Size the CUB final-combine scratch once; the query writes final_temp_bytes_
  // without touching d_tiles_/d_result_.
  CHECK_CUDA(cub::DeviceReduce::Reduce(nullptr, final_temp_bytes_, d_tiles_.get(), d_result_.get(), n_tiles_,
                                       peak_max{}, kPeakIdentity, ctx_.cuda_stream));
  d_final_temp_ = ctx_.alloc_ptr<std::byte>(core::narrow<std::int64_t>(std::max<std::size_t>(final_temp_bytes_, 1)));
}

void tiled_argmax_impl::reduce_tiles(core::span2d<const float> data, peak_criterion criterion, tile_range range)
{
  // Origin in (tile_x0, tile_y0), so each block still writes its absolute d_tiles slot.
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(core::narrow<unsigned>(range.tx1 - range.tx0 + 1), core::narrow<unsigned>(range.ty1 - range.ty0 + 1));
  kernel::tiled_argmax_reduce<<<grid, block, 0, ctx_.cuda_stream>>>(
      data.data_handle(), criterion, d_tiles_.get(), grid_.extents.extent(1), grid_.extents.extent(0), grid_.tile_size,
      grid_.tile_size, grid_.n_tiles_x, range.tx0, range.ty0);
  CHECK_CUDA(cudaGetLastError());
}

// A device-wide CUB reduce rather than one block: small tiles give ~10^5 slots, which one block would serialize.
// The op is associative and commutative with a deterministic tie-break, so CUB's block order is still exact.
peak tiled_argmax_impl::combine()
{
  CHECK_CUDA(cub::DeviceReduce::Reduce(d_final_temp_.get(), final_temp_bytes_, d_tiles_.get(), d_result_.get(),
                                       n_tiles_, peak_max{}, kPeakIdentity, ctx_.cuda_stream));

  peak h_result{};
  CHECK_CUDA(cudaMemcpyAsync(&h_result, d_result_.get(), sizeof(peak), cudaMemcpyDeviceToHost, ctx_.cuda_stream));
  ctx_.wait();

  return h_result;
}

}  // namespace fast_deconv::matrix
