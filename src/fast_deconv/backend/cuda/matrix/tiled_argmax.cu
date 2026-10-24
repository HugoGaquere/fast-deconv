#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <fast_deconv/matrix/tiled_argmax.hpp>
#include <fd_backend/core/launch.cuh>

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
    : ctx_(ctx), grid_(grid), d_tiles_(ctx.alloc_ptr<peak>(grid.n_tiles_x * grid.n_tiles_y))
{
}

void tiled_argmax_impl::reduce_tiles(core::span2d<const float> data, peak_criterion criterion, tile_range range)
{
  // Origin in (tile_x0, tile_y0), so each block still writes its absolute d_tiles slot.
  const auto cfg = core::launch_cfg::custom(
      dim3(core::narrow<unsigned>(range.tx1 - range.tx0 + 1), core::narrow<unsigned>(range.ty1 - range.ty0 + 1)),
      kThreadsPerBlock);
  core::launch(ctx_, cfg, kernel::tiled_argmax_reduce, data.data_handle(), criterion, d_tiles_.get(),
               grid_.extents.extent(1), grid_.extents.extent(0), grid_.tile_size, grid_.tile_size, grid_.n_tiles_x,
               range.tx0, range.ty0);
}

}  // namespace fast_deconv::matrix
