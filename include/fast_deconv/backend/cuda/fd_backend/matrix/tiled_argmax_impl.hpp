#pragma once

#include <cstddef>
#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/matrix/peak.hpp>
#include <fast_deconv/matrix/tile_grid.hpp>

namespace fast_deconv::matrix {

/**
 * @brief CUDA backend of tiled_argmax_ctx: the tile cache lives on the device, one block per tile.
 *
 * The constructor sizes the CUB final-combine scratch once.
 */
class tiled_argmax_impl {
 public:
  tiled_argmax_impl(const core::exec_ctx& ctx, const tile_grid& grid);

  /// Argmax of every tile in @p range under @p criterion, written into its cache slot.
  void reduce_tiles(core::span2d<const float> data, peak_criterion criterion, tile_range range);

  /// Combine every cached tile down to one result and read it back. Syncs the context.
  peak combine();

 private:
  const core::exec_ctx& ctx_;
  tile_grid grid_;
  std::int64_t n_tiles_;
  std::size_t final_temp_bytes_ = 0;
  core::owned_ptr<peak> d_tiles_;
  core::owned_ptr<peak> d_result_;
  core::owned_ptr<std::byte> d_final_temp_;
};

}  // namespace fast_deconv::matrix
