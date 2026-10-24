#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/matrix/tiled_argmax.hpp>
#include <fd_backend/core/reduce.hpp>
#include <stdexcept>

namespace fast_deconv::matrix {

namespace {

tile_grid make_grid(core::dims<2> extents, std::int64_t tile_size)
{
  if (extents.extent(0) <= 0 || extents.extent(1) <= 0 || tile_size <= 0)
    throw std::invalid_argument("tiled_argmax: image extents and tile_size must be > 0");

  return {.extents = extents,
          .tile_size = tile_size,
          .n_tiles_x = core::ceil_div(extents.extent(1), tile_size),
          .n_tiles_y = core::ceil_div(extents.extent(0), tile_size)};
}

// Cached slot i, as is.
struct tile_slot {
  const peak* tiles;

  FD_HOST_DEVICE peak operator()(std::int64_t i) const { return tiles[i]; }
};

}  // namespace

tiled_argmax_ctx::tiled_argmax_ctx(const core::exec_ctx& ctx, core::dims<2> extents, std::int64_t tile_size)
    : ctx_(ctx), grid_(make_grid(extents, tile_size)), impl_(ctx, grid_)
{
}

peak tiled_argmax_ctx::run(core::span2d<const float> data, peak_criterion criterion)
{
  FD_PROFILE_FN();
  assert(data.is_exhaustive());
  assert(data.extent(0) == grid_.extents.extent(0) && data.extent(1) == grid_.extents.extent(1));

  criterion_ = criterion;
  impl_.reduce_tiles(data, criterion_, {.tx0 = 0, .ty0 = 0, .tx1 = grid_.n_tiles_x - 1, .ty1 = grid_.n_tiles_y - 1});
  seeded_ = true;

  return combine_();
}

peak tiled_argmax_ctx::run_incremental(core::span2d<const float> data, std::int64_t peak_row, std::int64_t peak_col,
                                       std::int64_t foot_height, std::int64_t foot_width)
{
  FD_PROFILE_FN();
  assert(data.is_exhaustive());
  assert(data.extent(0) == grid_.extents.extent(0) && data.extent(1) == grid_.extents.extent(1));

  if (!seeded_) throw std::logic_error("tiled_argmax: run_incremental needs a prior run() to seed the tiles");

  const std::int64_t image_nrow = grid_.extents.extent(0);
  const std::int64_t image_ncol = grid_.extents.extent(1);

  // Dirtied pixel rectangle: the stamp is centered on the peak, half-open
  // [px0, px1) x [py0, py1), clamped to the image.
  const std::int64_t px0 = std::max<std::int64_t>(peak_col - foot_width / 2, 0);
  const std::int64_t py0 = std::max<std::int64_t>(peak_row - foot_height / 2, 0);
  const std::int64_t px1 = std::min(peak_col - foot_width / 2 + foot_width, image_ncol);
  const std::int64_t py1 = std::min(peak_row - foot_height / 2 + foot_height, image_nrow);
  if (px0 >= px1 || py0 >= py1)
    throw std::invalid_argument("tiled_argmax: incremental footprint lies outside the image");

  // Tile rectangle covering those pixels. The last dirtied pixel px1-1 lives in tile (px1-1)/tile_size.
  const std::int64_t ts = grid_.tile_size;
  impl_.reduce_tiles(data, criterion_,
                     {.tx0 = px0 / ts, .ty0 = py0 / ts, .tx1 = (px1 - 1) / ts, .ty1 = (py1 - 1) / ts});

  // Every cached slot is rescanned: the global winner can sit in any tile, and
  // after a subtraction last cycle's winner has dropped.
  return combine_();
}

peak tiled_argmax_ctx::combine_() const
{
  FD_PROFILE_FN();
  return core::reduce(ctx_, grid_.n_tiles_x * grid_.n_tiles_y, tile_slot{impl_.tiles()}, peak_max{}, kPeakIdentity);
}

}  // namespace fast_deconv::matrix
