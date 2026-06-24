#pragma once

#include <cub/cub.cuh>
#include <fast_deconv/core/resources.hpp>
#include <tuple>

namespace fast_deconv::matrix {

struct argmax_tile {
    int peak_index;   // row-major flat index into the FULL image (row * image_width + col)
    float peak_value;
};

// Caches one argmax_tile per tile of the image so that, after a clean
// subtraction only dirties a sub-region, the global argmax can be refreshed by
// recomputing the touched tiles instead of rescanning the whole image.
//
// Usage:
//   tiled_argmax_workspace ws{sr};
//   ws.image_width = W;  ws.image_height = H;
//   ws.tile_width  = 256; ws.tile_height = 256;
//   auto [val, idx] = matrix::argmax(ws, d_data);   // first call sizes + allocates
//   auto [val2, idx2] = matrix::argmax(ws, d_data); // reuses the same workspace
struct tiled_argmax_workspace {
  explicit tiled_argmax_workspace(const core::stream_resources& sr) : stream_res(sr) {}

  // Releases every device buffer allocated on first use. The async frees are
  // enqueued on the workspace's stream, so this must outlive nothing that still
  // reads d_tiles on that stream (it doesn't: argmax() syncs before returning).
  ~tiled_argmax_workspace()
  {
    stream_res.free_async(d_tiles);
    stream_res.free_async(d_final_temp);
    stream_res.free_async(d_result);
  }

  tiled_argmax_workspace(const tiled_argmax_workspace&) = delete;
  tiled_argmax_workspace& operator=(const tiled_argmax_workspace&) = delete;

  const core::stream_resources& stream_res;
  argmax_tile* d_tiles = nullptr;  // [n_tiles_y * n_tiles_x] cached per-tile maxima

  // CUB DeviceReduce final-combine scratch + result, sized/allocated on first use.
  void* d_final_temp = nullptr;
  size_t final_temp_bytes = 0;
  argmax_tile* d_result = nullptr;

  // Image + tile geometry. Filled by the caller before the first argmax call.
  size_t image_width = 0;
  size_t image_height = 0;
  size_t tile_width = 0;
  size_t tile_height = 0;

  // Derived on first use (see the launcher).
  size_t n_tiles_x = 0;
  size_t n_tiles_y = 0;
  size_t n_total_elements = 0;  // n_tiles_x * n_tiles_y
  bool tiles_initialized = false;
};

// void argmax_async(tiled_argmax_workspace& ws, const float* d_data);

// Blocking full-image tiled argmax over @p d_data (sized by ws.image_*).
// Returns {peak_value, flat_index}, flat_index row-major over the full image.
std::tuple<float, int> argmax(tiled_argmax_workspace& ws, const float* d_data);

// Blocking INCREMENTAL tiled argmax. Use after a clean subtraction has dirtied
// only a footprint rectangle of @p foot_height x @p foot_width centered on
// (peak_row, peak_col): it recomputes just the tiles overlapping that footprint
// (their cached maxima are now stale) and re-combines them against the still-
// valid cached maxima of every other tile.
//
// PRECONDITION: a prior argmax() call must have populated all tiles, and NOTHING
// outside the given footprint may have changed in @p d_data since the last call.
// Returns {peak_value, flat_index} over the full image, same as argmax().
std::tuple<float, int> argmax_incremental(tiled_argmax_workspace& ws, const float* d_data,
                                          int peak_row, int peak_col,
                                          int foot_height, int foot_width);

}  // namespace fast_deconv::matrix
