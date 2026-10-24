#pragma once

#include <concepts>
#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/matrix/peak.hpp>
#include <fast_deconv/matrix/tile_grid.hpp>
#include <utility>

// Resolved by the include path: CMake puts backend/${FAST_DECONV_BACKEND}
// on it, and every backend provides this file.
#include <fd_backend/matrix/tiled_argmax_impl.hpp>

namespace fast_deconv::matrix {

namespace detail {
/// What every backend's fd_backend/matrix/tiled_argmax_impl.hpp must provide.
template <typename T>
concept tiled_argmax_backend =
    std::constructible_from<T, const core::exec_ctx&, const tile_grid&> &&
    requires(T impl, core::span2d<const float> data, peak_criterion criterion, tile_range range) {
      { impl.reduce_tiles(data, criterion, range) } -> std::same_as<void>;
      { std::as_const(impl).tiles() } -> std::same_as<const peak*>;
    };

static_assert(tiled_argmax_backend<tiled_argmax_impl>,
              "fd_backend/matrix/tiled_argmax_impl.hpp must model tiled_argmax_backend");
}  // namespace detail

/**
 * @brief Caches one peak per tile of the image for incremental argmax refreshes.
 *
 * After a clean subtraction only dirties a sub-region, the global argmax can be refreshed by recomputing
 * the touched tiles instead of rescanning the whole image. The backend's tiled_argmax_impl holds the tile
 * cache; this layer validates the inputs, maps a footprint to the tiles it touches and combines the tiles.
 *
 * @code
 *   tiled_argmax_ctx tiled{ctx, mean_residual.extents(), 64};
 *   auto first = tiled.run(mean_residual);            // seeds every tile
 *   auto next  = tiled.run_incremental(mean_residual, row, col, fh, fw);
 * @endcode
 */
class tiled_argmax_ctx {
 public:
  /**
   * @param[in] ctx        Context the reductions run on. Must outlive this.
   * @param[in] extents    Image extents (nrow, ncol) every call must match.
   * @param[in] tile_size  Square tile edge in pixels.
   * @throws std::invalid_argument If an extent or @p tile_size is <= 0.
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
   * @pre A prior run() seeded every tile, and nothing outside the footprint changed in @p data since the
   *      previous run() or run_incremental().
   *
   * @param[in] data         Image, (nrow, ncol), matching the constructor's extents.
   * @param[in] peak_row     Footprint center row.
   * @param[in] peak_col     Footprint center column.
   * @param[in] foot_height  Footprint height in pixels.
   * @param[in] foot_width   Footprint width in pixels.
   * @return The refreshed global peak.
   * @throws std::logic_error If no run() seeded the tiles.
   * @throws std::invalid_argument If the footprint lies outside the image.
   */
  peak run_incremental(core::span2d<const float> data, std::int64_t peak_row, std::int64_t peak_col,
                       std::int64_t foot_height, std::int64_t foot_width);

 private:
  // The global peak over every cached tile. Syncs the context.
  peak combine_() const;

  const core::exec_ctx& ctx_;
  tile_grid grid_;
  tiled_argmax_impl impl_;
  peak_criterion criterion_{};
  bool seeded_ = false;
};

}  // namespace fast_deconv::matrix
