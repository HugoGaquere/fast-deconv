#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <fast_deconv/matrix/tiled_argmax.hpp>
#include <stdexcept>

namespace fast_deconv::matrix {

namespace {

// Max by value; ties go to the smaller index, matching the device reduction.
peak max_by_value(const peak& a, const peak& b)
{
  if (a.value > b.value) return a;
  if (b.value > a.value) return b;
  return (a.index <= b.index) ? a : b;
}

// Never wins: the largest index loses every tie against a real tile.
constexpr peak kReduceIdentity{-FLT_MAX, INT64_MAX};

}  // namespace

tiled_argmax_ctx::tiled_argmax_ctx(const core::exec_ctx& ctx, core::dims<2> extents, int tile_size)
    : ctx_(ctx), extents_(extents), tile_size_(tile_size)
{
  if (extents_.extent(0) <= 0 || extents_.extent(1) <= 0 || tile_size_ <= 0)
    throw std::invalid_argument("tiled_argmax: image extents and tile_size must be > 0");

  n_tiles_x_ = (extents_.extent(1) + tile_size_ - 1) / tile_size_;
  n_tiles_y_ = (extents_.extent(0) + tile_size_ - 1) / tile_size_;
  tiles_.assign(static_cast<std::size_t>(n_tiles_x_) * n_tiles_y_, kReduceIdentity);
}

void tiled_argmax_ctx::reduce_tile(core::span2d<float> data, int tile_x, int tile_y)
{
  const int image_nrow = extents_.extent(0);
  const int image_ncol = extents_.extent(1);
  const int row0 = tile_y * tile_size_;
  const int col0 = tile_x * tile_size_;
  const int row1 = std::min(row0 + tile_size_, image_nrow);
  const int col1 = std::min(col0 + tile_size_, image_ncol);

  const float* base = data.data_handle();
  peak best = kReduceIdentity;
  for (int r = row0; r < row1; r++) {
    // Rows are contiguous, so scan each one and keep the first maximum.
    const float* row = base + static_cast<std::int64_t>(r) * image_ncol;
    const float* found = std::max_element(row + col0, row + col1);
    const peak row_best{*found, static_cast<std::int64_t>(found - base)};
    best = max_by_value(best, row_best);
  }

  tiles_.at(static_cast<std::size_t>(tile_y) * n_tiles_x_ + tile_x) = best;
}

peak tiled_argmax_ctx::final_combine() const
{
  peak result = kReduceIdentity;
  for (const peak& t : tiles_) result = max_by_value(result, t);

  // An image that is entirely -inf never beats the identity; report index 0, as
  // the device path does, rather than leaking the sentinel.
  if (result.index == INT64_MAX) result.index = 0;
  return result;
}

peak tiled_argmax_ctx::run(core::span2d<float> data)
{
  assert(data.is_exhaustive());
  assert(data.extent(0) == extents_.extent(0) && data.extent(1) == extents_.extent(1));

  for (int ty = 0; ty < n_tiles_y_; ty++)
    for (int tx = 0; tx < n_tiles_x_; tx++) reduce_tile(data, tx, ty);

  seeded_ = true;
  return final_combine();
}

peak tiled_argmax_ctx::run_incremental(core::span2d<float> data, int peak_row, int peak_col, int foot_height,
                                       int foot_width)
{
  assert(data.is_exhaustive());
  assert(data.extent(0) == extents_.extent(0) && data.extent(1) == extents_.extent(1));

  if (!seeded_) throw std::logic_error("tiled_argmax: run_incremental needs a prior run() to seed the tiles");

  const int image_nrow = extents_.extent(0);
  const int image_ncol = extents_.extent(1);

  // Dirtied pixel rectangle: the stamp is centered on the peak, half-open and clamped.
  const int px0 = std::max(peak_col - foot_width / 2, 0);
  const int py0 = std::max(peak_row - foot_height / 2, 0);
  const int px1 = std::min(peak_col - foot_width / 2 + foot_width, image_ncol);
  const int py1 = std::min(peak_row - foot_height / 2 + foot_height, image_nrow);
  if (px0 >= px1 || py0 >= py1)
    throw std::invalid_argument("tiled_argmax: incremental footprint lies outside the image");

  // Tile rectangle covering those pixels (inclusive tile indices).
  const int tx0 = px0 / tile_size_;
  const int ty0 = py0 / tile_size_;
  const int tx1 = (px1 - 1) / tile_size_;
  const int ty1 = (py1 - 1) / tile_size_;

  for (int ty = ty0; ty <= ty1; ty++)
    for (int tx = tx0; tx <= tx1; tx++) reduce_tile(data, tx, ty);

  // Every cached slot is rescanned: the global winner can sit in any tile, and
  // after a subtraction last cycle's winner has dropped.
  return final_combine();
}

}  // namespace fast_deconv::matrix
