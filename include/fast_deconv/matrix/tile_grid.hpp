#pragma once

#include <cstdint>
#include <fast_deconv/core/dims.hpp>

namespace fast_deconv::matrix {

/// Square tiling of an image, row-major tile slots: the contract between tiled_argmax_ctx and tiled_argmax_impl.
struct tile_grid {
  core::dims<2> extents;  ///< image (nrow, ncol)
  std::int64_t tile_size;
  std::int64_t n_tiles_x;
  std::int64_t n_tiles_y;
};

/// Inclusive tile index rectangle [tx0, tx1] x [ty0, ty1].
struct tile_range {
  std::int64_t tx0;
  std::int64_t ty0;
  std::int64_t tx1;
  std::int64_t ty1;
};

}  // namespace fast_deconv::matrix
