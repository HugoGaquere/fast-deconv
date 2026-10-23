#pragma once

#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/matrix/peak.hpp>
#include <fast_deconv/matrix/tile_grid.hpp>
#include <vector>

namespace fast_deconv::matrix {

/// Host backend of tiled_argmax_ctx: the tile cache is a host vector, scanned with OpenMP.
class tiled_argmax_impl {
 public:
  tiled_argmax_impl(const core::exec_ctx& ctx, const tile_grid& grid);

  /// Argmax of every tile in @p range under @p criterion, written into its cache slot.
  void reduce_tiles(core::span2d<const float> data, peak_criterion criterion, tile_range range);

  /// Combine every cached tile down to one result.
  peak combine() const;

 private:
  tile_grid grid_;
  std::vector<peak> tiles_;
};

}  // namespace fast_deconv::matrix
