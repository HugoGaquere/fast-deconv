#pragma once

#include <cstddef>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/matrix/peak.hpp>
#include <vector>

namespace fast_deconv::matrix {

/// Caches one peak per tile of the image so that, after a clean subtraction
/// only dirties a sub-region, the global argmax can be refreshed by
/// recomputing the touched tiles instead of rescanning the whole image. At
/// production sizes (~4e8 pixels) that is the difference between gigabytes of
/// memory traffic per minor cycle and a few megabytes.
///
///   tiled_argmax_ctx tiled{ctx, mean_residual.extents(), 64};
///   auto first = tiled.run(mean_residual);            // seeds every tile
///   auto next  = tiled.run_incremental(mean_residual, row, col, fh, fw);
class tiled_argmax_ctx {
 public:
  /// @param extents    image extents (nrow, ncol) every call must match.
  /// @param tile_size  square tile edge in pixels.
  tiled_argmax_ctx(const core::exec_ctx& ctx, core::dims<2> extents, int tile_size);

  tiled_argmax_ctx(const tiled_argmax_ctx&) = delete;
  tiled_argmax_ctx& operator=(const tiled_argmax_ctx&) = delete;

  /// Full pass: recompute every tile, then combine. Seeds the cache.
  peak run(core::span2d<float> data);

  /// Incremental pass: recompute only the tiles overlapping the
  /// @p foot_height x @p foot_width rectangle centered on (@p peak_row,
  /// @p peak_col), then re-combine against the still-valid cached tiles.
  ///
  /// PRECONDITION: a prior run() seeded every tile, and nothing outside the
  /// footprint changed in @p data since that call.
  peak run_incremental(core::span2d<float> data, int peak_row, int peak_col, int foot_height, int foot_width);

  core::dims<2> extents() const { return extents_; }
  int tile_size() const { return tile_size_; }

 private:
  /// Argmax over one tile, written into its cache slot.
  void reduce_tile(core::span2d<float> data, int tile_x, int tile_y);

  /// Combine every cached tile down to one result.
  peak final_combine() const;

  const core::exec_ctx& ctx_;
  core::dims<2> extents_;
  int tile_size_;
  int n_tiles_x_;
  int n_tiles_y_;
  std::vector<peak> tiles_;
  bool seeded_ = false;
};

}  // namespace fast_deconv::matrix
