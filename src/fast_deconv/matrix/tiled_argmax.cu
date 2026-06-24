#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <stdexcept>
#include <tuple>

#include <fast_deconv/matrix/tiled_argmax.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace {
// Threads per block for the per-tile reduction. A block reduces its whole tile
// with a grid-stride loop, so this is INDEPENDENT of the tile size: pick it for
// occupancy, not to "cover" the tile. 256 is a safe default.
constexpr int kThreadsPerBlock = 256;
}  // namespace

namespace fast_deconv::kernel {

using fast_deconv::matrix::argmax_tile;

  struct MaxByValue {
    __device__ __forceinline__
    argmax_tile operator()(const argmax_tile& a, const argmax_tile& b) const
    {
      if (a.peak_value > b.peak_value) return a;
      if (b.peak_value > a.peak_value) return b;
      return (a.peak_index <= b.peak_index) ? a : b;   // tie on value -> smaller index
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
// Each block must end by writing exactly ONE argmax_tile into
//   d_tiles[tile_y * n_tiles_x + tile_x]
// where (peak_value, peak_index) is the max over its tile and peak_index is the
// flat index into the FULL image (gy * image_width + gx).
__global__ void tiled_argmax_reduce(const float* data, argmax_tile* d_tiles,
                                    int image_width, int image_height, int tile_width, int tile_height,
                                    int n_tiles_x, int tile_x0, int tile_y0)
{
  // Identify the tile this block owns
  const int tile_x = tile_x0 + blockIdx.x;
  const int tile_y = tile_y0 + blockIdx.y;
  // Locate the origin of this block
  const int ox = tile_x * tile_width;
  const int oy = tile_y * tile_height;
  // Clamp the tile extent to the image
  const int ex = min(ox+tile_width, image_width);
  const int ey = min(oy+tile_height, image_height);
  // Actual tile width and height
  const int tw = ex - ox, th = ey - oy;

  // Precompute col and row step
  const int step = blockDim.x;
  const int dcol = step % tw;
  const int drow = step / tw;
  int col = threadIdx.x % tw;
  int row = threadIdx.x / tw;

  // Reduce to a best per thread
  argmax_tile best {0, -FLT_MAX};
  while (row < th) {
    const int g = (oy+row) * image_width + (ox+col);
    const float v = data[g];
    if (v > best.peak_value) best = {g, v};
    row += drow; col += dcol;
    if (col >= tw) {col -= tw; row++;}
  }

  // Reduce to one winner per block
  using BlockReduce = cub::BlockReduce<argmax_tile, kThreadsPerBlock>;
  __shared__ typename BlockReduce::TempStorage temp;
  argmax_tile winner = BlockReduce(temp).Reduce(best, MaxByValue{});

  if(threadIdx.x == 0) d_tiles[tile_y * n_tiles_x + tile_x] = winner;
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::matrix {

namespace {

// Identity for the max-by-value reduction: a value of -FLT_MAX never wins, and the
// largest possible index loses every tie, so this entry is overridden by any real
// tile (it only survives an all-(-FLT_MAX) input, which the image never produces).
constexpr argmax_tile kReduceIdentity{INT_MAX, -FLT_MAX};

// Combine all cached per-tile maxima in ws.d_tiles down to one global result and
// read it back to the host. Shared by the full and incremental entry points: both
// must rescan EVERY cached slot, because the global winner can sit in any tile
// (and after a clean subtraction last cycle's winner has dropped).
//
// A full device-wide CUB reduction (vs a single-block kernel) keeps the combine
// parallel even when n_tiles is large (small tiles -> ~10^5 slots), which is where
// a one-block reduce would serialize. The op is associative + commutative with a
// deterministic tie-break, so CUB's nondeterministic block order is still exact.
std::tuple<float, int> final_combine_blocking(tiled_argmax_workspace& ws)
{
  const core::stream_resources& sr = ws.stream_res;

  CHECK_CUDA(cub::DeviceReduce::Reduce(ws.d_final_temp, ws.final_temp_bytes, ws.d_tiles, ws.d_result,
                                       static_cast<int>(ws.n_total_elements), kernel::MaxByValue{},
                                       kReduceIdentity, sr.cuda_stream));

  argmax_tile h_result{};
  CHECK_CUDA(
      cudaMemcpyAsync(&h_result, ws.d_result, sizeof(argmax_tile), cudaMemcpyDeviceToHost, sr.cuda_stream));
  sr.sync();

  return {h_result.peak_value, h_result.peak_index};
}

}  // namespace

// Host launcher: per-tile reduce over the whole grid, then the CUB device-wide
// final combine. First call sizes + allocates the workspace buffers.
std::tuple<float, int> argmax(tiled_argmax_workspace& ws, const float* d_data)
{
  const core::stream_resources& sr = ws.stream_res;

  // ---- One-time geometry + buffer init (reused across subsequent calls) ----
  if (!ws.tiles_initialized) {
    if (ws.image_width == 0 || ws.image_height == 0 || ws.tile_width == 0 || ws.tile_height == 0)
      throw std::invalid_argument("tiled_argmax: image_/tile_ geometry must be set before argmax()");

    ws.n_tiles_x = (ws.image_width + ws.tile_width - 1) / ws.tile_width;
    ws.n_tiles_y = (ws.image_height + ws.tile_height - 1) / ws.tile_height;
    ws.n_total_elements = ws.n_tiles_x * ws.n_tiles_y;
    ws.d_tiles = sr.alloc_async<argmax_tile>(ws.n_total_elements);
    ws.d_result = sr.alloc_async<argmax_tile>(1);

    // Size the CUB final-combine scratch once (n_total_elements is now fixed). The
    // query writes final_temp_bytes without touching d_tiles/d_result.
    CHECK_CUDA(cub::DeviceReduce::Reduce(nullptr, ws.final_temp_bytes, ws.d_tiles, ws.d_result,
                                         static_cast<int>(ws.n_total_elements), kernel::MaxByValue{},
                                         kReduceIdentity, sr.cuda_stream));
    ws.d_final_temp = sr.alloc_async(std::max<size_t>(ws.final_temp_bytes, 1));
    ws.tiles_initialized = true;
  }

  // ---- Full pass: one block per tile over the whole image ----
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned>(ws.n_tiles_x), static_cast<unsigned>(ws.n_tiles_y));
  kernel::tiled_argmax_reduce<<<grid, block, 0, sr.cuda_stream>>>(
      d_data, ws.d_tiles, static_cast<int>(ws.image_width), static_cast<int>(ws.image_height),
      static_cast<int>(ws.tile_width), static_cast<int>(ws.tile_height), static_cast<int>(ws.n_tiles_x),
      /*tile_x0=*/0, /*tile_y0=*/0);
  CHECK_CUDA(cudaGetLastError());

  // ---- Final combine over all tiles -> single result ----
  return final_combine_blocking(ws);
}

// Incremental launcher — recompute only the tiles the clean subtraction dirtied,
// then re-combine against the untouched cached tiles. The per-tile kernel already
// supports a partial grid via (tile_x0, tile_y0); here we just translate the
// dirtied pixel footprint into that tile sub-grid.
std::tuple<float, int> argmax_incremental(tiled_argmax_workspace& ws, const float* d_data,
                                          int peak_row, int peak_col, int foot_height, int foot_width)
{
  const core::stream_resources& sr = ws.stream_res;

  if (!ws.tiles_initialized)
    throw std::logic_error("tiled_argmax: argmax_incremental needs a prior full argmax() to seed the tiles");

  const int iw = static_cast<int>(ws.image_width);
  const int ih = static_cast<int>(ws.image_height);
  const int tw = static_cast<int>(ws.tile_width);
  const int th = static_cast<int>(ws.tile_height);

  // ---- Dirtied pixel rectangle: the stamp is centered on the peak, half-open
  //      [px0, px1) x [py0, py1), clamped to the image. ----
  const int px0 = std::max(peak_col - foot_width / 2, 0);
  const int py0 = std::max(peak_row - foot_height / 2, 0);
  const int px1 = std::min(peak_col - foot_width / 2 + foot_width, iw);
  const int py1 = std::min(peak_row - foot_height / 2 + foot_height, ih);
  if (px0 >= px1 || py0 >= py1)
    throw std::invalid_argument("tiled_argmax: incremental footprint lies outside the image");

  // ---- Tile rectangle covering those pixels (inclusive tile indices). The last
  //      dirtied pixel px1-1 lives in tile (px1-1)/tile_width. ----
  const int tx0 = px0 / tw;
  const int ty0 = py0 / th;
  const int tx1 = (px1 - 1) / tw;
  const int ty1 = (py1 - 1) / th;

  // ---- Relaunch kernel 1 over just the dirty tile sub-grid; its origin goes in
  //      (tile_x0, tile_y0) so each block still writes its absolute d_tiles slot. ----
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned>(tx1 - tx0 + 1), static_cast<unsigned>(ty1 - ty0 + 1));
  kernel::tiled_argmax_reduce<<<grid, block, 0, sr.cuda_stream>>>(
      d_data, ws.d_tiles, iw, ih, tw, th, static_cast<int>(ws.n_tiles_x),
      /*tile_x0=*/tx0, /*tile_y0=*/ty0);
  CHECK_CUDA(cudaGetLastError());

  // ---- Final combine over ALL tiles (dirty + cached) -> single result ----
  return final_combine_blocking(ws);
}

}  // namespace fast_deconv::matrix
