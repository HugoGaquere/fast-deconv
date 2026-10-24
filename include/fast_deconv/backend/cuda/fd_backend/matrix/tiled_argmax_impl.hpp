#pragma once

#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/matrix/peak.hpp>
#include <fast_deconv/matrix/tile_grid.hpp>

namespace fast_deconv::matrix {

/**
 * @brief CUDA backend of tiled_argmax_ctx: the tile cache lives on the device, one block per tile.
 */
class tiled_argmax_impl {
 public:
  tiled_argmax_impl(const core::exec_ctx& ctx, const tile_grid& grid);

  /// Argmax of every tile in @p range under @p criterion, written into its cache slot.
  void reduce_tiles(core::span2d<const float> data, peak_criterion criterion, tile_range range);

  /// The device cache, one peak per tile in row-major tile order.
  const peak* tiles() const { return d_tiles_.get(); }

 private:
  const core::exec_ctx& ctx_;
  tile_grid grid_;
  core::owned_ptr<peak> d_tiles_;
};

}  // namespace fast_deconv::matrix
