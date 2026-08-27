#pragma once

#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/matrix/peak.hpp>

namespace fast_deconv::matrix {

/// Tiling exists to avoid rescanning device memory; a host argmax is a linear
/// scan, so the cache buys nothing here and the type only keeps the call sites
/// identical across backends.
class tiled_argmax_ctx {
 public:
  tiled_argmax_ctx(const core::exec_ctx& ctx, core::dims<2> extents, int tile_size)
      : ctx_(ctx), extents_(extents), tile_size_(tile_size)
  {
  }

  tiled_argmax_ctx(const tiled_argmax_ctx&) = delete;
  tiled_argmax_ctx& operator=(const tiled_argmax_ctx&) = delete;

  peak run(core::span2d<float> data);
  peak run_incremental(core::span2d<float> data, int peak_row, int peak_col, int foot_height, int foot_width);

  core::dims<2> extents() const { return extents_; }
  int tile_size() const { return tile_size_; }

 private:
  const core::exec_ctx& ctx_;
  core::dims<2> extents_;
  int tile_size_;
};

}  // namespace fast_deconv::matrix
