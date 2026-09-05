#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>
#include <fast_deconv/matrix/tiled_argmax.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <stdexcept>

namespace {
// Threads per block for the per-tile reduction. A block reduces its whole tile
// with a grid-stride loop, so this is INDEPENDENT of the tile size: pick it for
// occupancy, not to "cover" the tile. 256 is a safe default.
constexpr int kThreadsPerBlock = 256;
}  // namespace

namespace fast_deconv::kernel {

using fast_deconv::matrix::peak;

struct MaxByValue {
  __device__ __forceinline__ peak operator()(const peak& a, const peak& b) const
  {
    if (a.value > b.value) return a;
    if (b.value > a.value) return b;
    return (a.index <= b.index) ? a : b;  // tie on value -> smaller index
  }
};

// ===========================================================================
// KERNEL 1 — per-tile argmax.   One block == one tile.
// ===========================================================================
// Launched with a 2D grid covering the tile rectangle you want to (re)compute.
// For a FULL pass the launcher uses grid = (n_tiles_x, n_tiles_y) and
// tile_x0 = tile_y0 = 0. For a future INCREMENTAL pass you would launch a
// smaller grid over just the dirty tile rectangle and pass its origin in
// (tile_x0, tile_y0) — the body below is written so the same code serves both.
//
// Each block must end by writing exactly ONE peak into
//   d_tiles[tile_y * n_tiles_x + tile_x]
// where (value, index) is the max over its tile and index is the flat index into
// the FULL image (gy * image_width + gx).
__global__ void tiled_argmax_reduce(const float* data, peak* d_tiles, int image_width, int image_height, int tile_width,
                                    int tile_height, int n_tiles_x, int tile_x0, int tile_y0)
{
  // Identify the tile this block owns
  const int tile_x = tile_x0 + blockIdx.x;
  const int tile_y = tile_y0 + blockIdx.y;
  // Locate the origin of this block
  const int ox = tile_x * tile_width;
  const int oy = tile_y * tile_height;
  // Clamp the tile extent to the image
  const int ex = min(ox + tile_width, image_width);
  const int ey = min(oy + tile_height, image_height);
  // Actual tile width and height
  const int tw = ex - ox, th = ey - oy;

  // Precompute col and row step
  const int step = blockDim.x;
  const int dcol = step % tw;
  const int drow = step / tw;
  int col = threadIdx.x % tw;
  int row = threadIdx.x / tw;

  // Reduce to a best per thread
  peak best{-INFINITY, 0};
  while (row < th) {
    const int g = (oy + row) * image_width + (ox + col);
    const float v = data[g];
    if (v > best.value) best = {v, g};
    row += drow;
    col += dcol;
    if (col >= tw) {
      col -= tw;
      row++;
    }
  }

  // Reduce to one winner per block
  using BlockReduce = cub::BlockReduce<peak, kThreadsPerBlock>;
  __shared__ typename BlockReduce::TempStorage temp;
  peak winner = BlockReduce(temp).Reduce(best, MaxByValue{});

  if (threadIdx.x == 0) d_tiles[tile_y * n_tiles_x + tile_x] = winner;
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::matrix {

namespace {

// Identity for the max-by-value reduction: -inf is the masked-pixel fill, so it
// ties rather than wins, and the largest possible index loses that tie. Any real
// tile overrides it.
constexpr peak kReduceIdentity{-INFINITY, INT64_MAX};

}  // namespace

tiled_argmax_ctx::tiled_argmax_ctx(const core::exec_ctx& ctx, core::dims<2> extents, int tile_size)
    : ctx_(ctx), extents_(extents), tile_size_(tile_size)
{
  if (extents_.extent(0) <= 0 || extents_.extent(1) <= 0 || tile_size_ <= 0)
    throw std::invalid_argument("tiled_argmax: image extents and tile_size must be > 0");

  n_tiles_x_ = (extents_.extent(1) + tile_size_ - 1) / tile_size_;
  n_tiles_y_ = (extents_.extent(0) + tile_size_ - 1) / tile_size_;
  n_tiles_ = static_cast<std::size_t>(n_tiles_x_) * static_cast<std::size_t>(n_tiles_y_);

  d_tiles_ = ctx_.alloc_ptr_async<peak>(n_tiles_);
  d_result_ = ctx_.alloc_ptr_async<peak>(1);

  // Size the CUB final-combine scratch once; the query writes final_temp_bytes_
  // without touching d_tiles_/d_result_.
  CHECK_CUDA(cub::DeviceReduce::Reduce(nullptr, final_temp_bytes_, d_tiles_.get(), d_result_.get(),
                                       static_cast<int>(n_tiles_), kernel::MaxByValue{}, kReduceIdentity,
                                       ctx_.cuda_stream));
  d_final_temp_ = ctx_.alloc_ptr_async<std::byte>(std::max<std::size_t>(final_temp_bytes_, 1));
}

// Combine all cached per-tile maxima down to one global result and read it back.
// Shared by the full and incremental entry points: both must rescan EVERY cached
// slot, because the global winner can sit in any tile (and after a clean
// subtraction last cycle's winner has dropped).
//
// A full device-wide CUB reduction (vs a single-block kernel) keeps the combine
// parallel even when n_tiles is large (small tiles -> ~10^5 slots), which is where
// a one-block reduce would serialize. The op is associative + commutative with a
// deterministic tie-break, so CUB's nondeterministic block order is still exact.
peak tiled_argmax_ctx::final_combine()
{
  CHECK_CUDA(cub::DeviceReduce::Reduce(d_final_temp_.get(), final_temp_bytes_, d_tiles_.get(), d_result_.get(),
                                       static_cast<int>(n_tiles_), kernel::MaxByValue{}, kReduceIdentity,
                                       ctx_.cuda_stream));

  peak h_result{};
  CHECK_CUDA(cudaMemcpyAsync(&h_result, d_result_.get(), sizeof(peak), cudaMemcpyDeviceToHost, ctx_.cuda_stream));
  ctx_.wait();

  return h_result;
}

// Per-tile reduce over the whole grid, then the CUB device-wide final combine.
peak tiled_argmax_ctx::run(core::span2d<float> data)
{
  assert(data.is_exhaustive());
  assert(data.extent(0) == extents_.extent(0) && data.extent(1) == extents_.extent(1));

  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned>(n_tiles_x_), static_cast<unsigned>(n_tiles_y_));
  kernel::tiled_argmax_reduce<<<grid, block, 0, ctx_.cuda_stream>>>(data.data_handle(), d_tiles_.get(),
                                                                    extents_.extent(1), extents_.extent(0), tile_size_,
                                                                    tile_size_, n_tiles_x_, /*tile_x0=*/0,
                                                                    /*tile_y0=*/0);
  CHECK_CUDA(cudaGetLastError());
  seeded_ = true;

  return final_combine();
}

// Recompute only the tiles the clean subtraction dirtied, then re-combine against
// the untouched cached tiles. The per-tile kernel already supports a partial grid
// via (tile_x0, tile_y0); here we just translate the dirtied pixel footprint into
// that tile sub-grid.
peak tiled_argmax_ctx::run_incremental(core::span2d<float> data, int peak_row, int peak_col, int foot_height,
                                       int foot_width)
{
  assert(data.is_exhaustive());
  assert(data.extent(0) == extents_.extent(0) && data.extent(1) == extents_.extent(1));

  if (!seeded_) throw std::logic_error("tiled_argmax: run_incremental needs a prior run() to seed the tiles");

  const int iw = extents_.extent(1);
  const int ih = extents_.extent(0);

  // ---- Dirtied pixel rectangle: the stamp is centered on the peak, half-open
  //      [px0, px1) x [py0, py1), clamped to the image. ----
  const int px0 = std::max(peak_col - foot_width / 2, 0);
  const int py0 = std::max(peak_row - foot_height / 2, 0);
  const int px1 = std::min(peak_col - foot_width / 2 + foot_width, iw);
  const int py1 = std::min(peak_row - foot_height / 2 + foot_height, ih);
  if (px0 >= px1 || py0 >= py1)
    throw std::invalid_argument("tiled_argmax: incremental footprint lies outside the image");

  // ---- Tile rectangle covering those pixels (inclusive tile indices). The last
  //      dirtied pixel px1-1 lives in tile (px1-1)/tile_size. ----
  const int tx0 = px0 / tile_size_;
  const int ty0 = py0 / tile_size_;
  const int tx1 = (px1 - 1) / tile_size_;
  const int ty1 = (py1 - 1) / tile_size_;

  // ---- Relaunch kernel 1 over just the dirty tile sub-grid; its origin goes in
  //      (tile_x0, tile_y0) so each block still writes its absolute d_tiles slot. ----
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned>(tx1 - tx0 + 1), static_cast<unsigned>(ty1 - ty0 + 1));
  kernel::tiled_argmax_reduce<<<grid, block, 0, ctx_.cuda_stream>>>(data.data_handle(), d_tiles_.get(), iw, ih,
                                                                    tile_size_, tile_size_, n_tiles_x_,
                                                                    /*tile_x0=*/tx0, /*tile_y0=*/ty0);
  CHECK_CUDA(cudaGetLastError());

  // ---- Final combine over ALL tiles (dirty + cached) -> single result ----
  return final_combine();
}

}  // namespace fast_deconv::matrix
