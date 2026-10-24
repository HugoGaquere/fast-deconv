#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/matrix/tiled_argmax.hpp>

namespace fast_deconv::matrix {

namespace {

// A tile scan is a few microseconds, so a handful of them already pays the fork.
constexpr std::int64_t kMinParallelTiles = 8;

// Argmax of tile (tile_x, tile_y) of @p grid under @p criterion.
peak reduce_tile(core::span2d<const float> data, peak_criterion criterion, const tile_grid& grid, std::int64_t tile_x,
                 std::int64_t tile_y)
{
  const std::int64_t image_nrow = grid.extents.extent(0);
  const std::int64_t image_ncol = grid.extents.extent(1);
  const std::int64_t row0 = tile_y * grid.tile_size;
  const std::int64_t col0 = tile_x * grid.tile_size;
  const std::int64_t row1 = std::min(row0 + grid.tile_size, image_nrow);
  const std::int64_t col1 = std::min(col0 + grid.tile_size, image_ncol);

  const float* base = data.data_handle();
  peak best = kPeakIdentity;
  for (std::int64_t r = row0; r < row1; r++) {
    const std::int64_t row_offset = r * image_ncol;
    for (std::int64_t c = col0; c < col1; c++) {
      const std::int64_t i = row_offset + c;
      const float v = criterion(base[i], i);
      if (v > best.value) best = {.index = i, .value = v, .signed_value = base[i]};  // strict >: first maximum kept
    }
  }
  return best;
}

}  // namespace

tiled_argmax_impl::tiled_argmax_impl(const core::exec_ctx& /*ctx*/, const tile_grid& grid)
    : grid_(grid), tiles_(core::narrow<std::size_t>(grid.n_tiles_x * grid.n_tiles_y), kPeakIdentity)
{
}

void tiled_argmax_impl::reduce_tiles(core::span2d<const float> data, peak_criterion criterion, tile_range range)
{
  FD_PROFILE_FN();
#pragma omp parallel for collapse(2) if ((range.ty1 - range.ty0 + 1) * (range.tx1 - range.tx0 + 1) > kMinParallelTiles)
  for (std::int64_t ty = range.ty0; ty <= range.ty1; ty++)
    for (std::int64_t tx = range.tx0; tx <= range.tx1; tx++)
      tiles_.at(core::narrow<std::size_t>(ty * grid_.n_tiles_x + tx)) = reduce_tile(data, criterion, grid_, tx, ty);
}

}  // namespace fast_deconv::matrix
