#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/matrix/tiled_argmax.hpp>
#include <iterator>
#include <stdexcept>

#include "../detail/peak_reduce.hpp"

namespace fast_deconv::matrix {

namespace {

using detail::max_by_value;

// A tile scan is a few microseconds, so a handful of them already pays the fork.
constexpr std::int64_t kMinParallelTiles = 8;
// The combine is one compare per tile, so it needs far more of them to pay off.
constexpr std::int64_t kMinParallelCombine = 8192;

}  // namespace

tiled_argmax_ctx::tiled_argmax_ctx(const core::exec_ctx& ctx, core::dims<2> extents, std::int64_t tile_size)
    : ctx_(ctx), extents_(extents), tile_size_(tile_size)
{
  FD_PROFILE_FN();
  if (extents_.extent(0) <= 0 || extents_.extent(1) <= 0 || tile_size_ <= 0)
    throw std::invalid_argument("tiled_argmax: image extents and tile_size must be > 0");

  n_tiles_x_ = core::ceil_div(extents_.extent(1), tile_size_);
  n_tiles_y_ = core::ceil_div(extents_.extent(0), tile_size_);
  tiles_.assign(core::narrow<std::size_t>(n_tiles_x_ * n_tiles_y_), detail::kPeakIdentity);
}

void tiled_argmax_ctx::reduce_tile(core::span2d<const float> data, std::int64_t tile_x, std::int64_t tile_y)
{
  const std::int64_t image_nrow = extents_.extent(0);
  const std::int64_t image_ncol = extents_.extent(1);
  const std::int64_t row0 = tile_y * tile_size_;
  const std::int64_t col0 = tile_x * tile_size_;
  const std::int64_t row1 = std::min(row0 + tile_size_, image_nrow);
  const std::int64_t col1 = std::min(col0 + tile_size_, image_ncol);

  const float* base = data.data_handle();
  peak best = detail::kPeakIdentity;
  for (std::int64_t r = row0; r < row1; r++) {
    const std::int64_t row_offset = r * image_ncol;
    for (std::int64_t c = col0; c < col1; c++) {
      const std::int64_t i = row_offset + c;
      const float v = criterion_(base[i], i);
      if (v > best.value) best = {.index = i, .value = v, .signed_value = base[i]};  // strict >: first maximum kept
    }
  }

  tiles_.at(core::narrow<std::size_t>(tile_y * n_tiles_x_ + tile_x)) = best;
}

peak tiled_argmax_ctx::final_combine() const
{
  FD_PROFILE_FN();
  peak result = detail::kPeakIdentity;
  const std::int64_t n_tiles = std::ssize(tiles_);
#pragma omp parallel for reduction(peak_max : result) if (n_tiles > kMinParallelCombine)
  for (std::int64_t i = 0; i < n_tiles; i++) result = max_by_value(result, tiles_.at(core::narrow<std::size_t>(i)));

  return result;
}

peak tiled_argmax_ctx::run(core::span2d<const float> data, peak_criterion criterion)
{
  FD_PROFILE_FN();
  assert(data.is_exhaustive());
  assert(data.extent(0) == extents_.extent(0) && data.extent(1) == extents_.extent(1));

  criterion_ = criterion;
#pragma omp parallel for collapse(2) if (n_tiles_y_ * n_tiles_x_ > kMinParallelTiles)
  for (std::int64_t ty = 0; ty < n_tiles_y_; ty++)
    for (std::int64_t tx = 0; tx < n_tiles_x_; tx++) reduce_tile(data, tx, ty);

  seeded_ = true;
  return final_combine();
}

peak tiled_argmax_ctx::run_incremental(core::span2d<const float> data, std::int64_t peak_row, std::int64_t peak_col,
                                       std::int64_t foot_height, std::int64_t foot_width)
{
  FD_PROFILE_FN();
  assert(data.is_exhaustive());
  assert(data.extent(0) == extents_.extent(0) && data.extent(1) == extents_.extent(1));

  if (!seeded_) throw std::logic_error("tiled_argmax: run_incremental needs a prior run() to seed the tiles");

  const std::int64_t image_nrow = extents_.extent(0);
  const std::int64_t image_ncol = extents_.extent(1);

  // Dirtied pixel rectangle: the stamp is centered on the peak, half-open and clamped.
  const std::int64_t px0 = std::max<std::int64_t>(peak_col - foot_width / 2, 0);
  const std::int64_t py0 = std::max<std::int64_t>(peak_row - foot_height / 2, 0);
  const std::int64_t px1 = std::min(peak_col - foot_width / 2 + foot_width, image_ncol);
  const std::int64_t py1 = std::min(peak_row - foot_height / 2 + foot_height, image_nrow);
  if (px0 >= px1 || py0 >= py1)
    throw std::invalid_argument("tiled_argmax: incremental footprint lies outside the image");

  // Tile rectangle covering those pixels (inclusive tile indices).
  const std::int64_t tx0 = px0 / tile_size_;
  const std::int64_t ty0 = py0 / tile_size_;
  const std::int64_t tx1 = (px1 - 1) / tile_size_;
  const std::int64_t ty1 = (py1 - 1) / tile_size_;

#pragma omp parallel for collapse(2) if ((ty1 - ty0 + 1) * (tx1 - tx0 + 1) > kMinParallelTiles)
  for (std::int64_t ty = ty0; ty <= ty1; ty++)
    for (std::int64_t tx = tx0; tx <= tx1; tx++) reduce_tile(data, tx, ty);

  // Every cached slot is rescanned: the global winner can sit in any tile, and
  // after a subtraction last cycle's winner has dropped.
  return final_combine();
}

}  // namespace fast_deconv::matrix
