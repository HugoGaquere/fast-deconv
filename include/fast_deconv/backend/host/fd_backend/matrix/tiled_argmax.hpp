#pragma once

#include <cstddef>
#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/matrix/peak.hpp>
#include <vector>

namespace fast_deconv::matrix {

/**
 * @brief Caches one peak per tile of the image for incremental argmax refreshes.
 *
 * After a clean subtraction only dirties a sub-region, the global argmax can be refreshed by recomputing
 * the touched tiles instead of rescanning the whole image.
 */
class tiled_argmax_ctx {
 public:
  /**
   * @param[in] ctx        Lane the reductions run on. Must outlive this.
   * @param[in] extents    Image extents (nrow, ncol) every call must match.
   * @param[in] tile_size  Square tile edge in pixels.
   */
  tiled_argmax_ctx(const core::exec_ctx& ctx, core::dims<2> extents, std::int64_t tile_size);

  tiled_argmax_ctx(const tiled_argmax_ctx&) = delete;
  tiled_argmax_ctx& operator=(const tiled_argmax_ctx&) = delete;

  /// Full pass: recompute every tile under @p criterion, then combine. Seeds the cache and the criterion.
  peak run(core::span2d<const float> data, peak_criterion criterion = {});

  /**
   * @brief Incremental pass, under the criterion of the last run().
   *
   * Recomputes only the tiles overlapping the footprint, then re-combines against the still-valid cached
   * tiles.
   *
   * @pre A prior run() seeded every tile, and nothing outside the footprint changed in @p data since that call.
   *
   * @param[in] data         Image, (nrow, ncol), matching extents().
   * @param[in] peak_row     Footprint center row.
   * @param[in] peak_col     Footprint center column.
   * @param[in] foot_height  Footprint height in pixels.
   * @param[in] foot_width   Footprint width in pixels.
   * @return The refreshed global peak.
   */
  peak run_incremental(core::span2d<const float> data, std::int64_t peak_row, std::int64_t peak_col,
                       std::int64_t foot_height, std::int64_t foot_width);

  core::dims<2> extents() const { return extents_; }
  std::int64_t tile_size() const { return tile_size_; }

 private:
  /// Argmax over one tile, written into its cache slot.
  void reduce_tile(core::span2d<const float> data, std::int64_t tile_x, std::int64_t tile_y);

  /// Combine every cached tile down to one result.
  peak final_combine() const;

  const core::exec_ctx& ctx_;
  core::dims<2> extents_;
  std::int64_t tile_size_;
  std::int64_t n_tiles_x_;
  std::int64_t n_tiles_y_;
  std::vector<peak> tiles_;
  peak_criterion criterion_{};
  bool seeded_ = false;
};

}  // namespace fast_deconv::matrix
