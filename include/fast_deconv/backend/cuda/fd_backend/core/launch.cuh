#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <fast_deconv/core/dims.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <source_location>
#include <stdexcept>
#include <string>
#include <utility>

namespace fast_deconv::core {

/// Block size of 1D launches: one thread per element.
inline constexpr unsigned kBlock1d = 256;

/// Block shape of 2D and 3D launches: one thread per pixel, 32 wide so a warp reads one contiguous row segment.
inline constexpr dim3 kBlock2d(32, 8);

/**
 * @brief Grid and block of one kernel launch, plus the caller's location.
 *
 * Every constructor records where it is called, so a failed launch reports the call site, not this header.
 */
struct launch_cfg {
  dim3 grid;                 ///< Blocks per dimension.
  dim3 block;                ///< Threads per block.
  std::source_location loc;  ///< Where this config was built.

  /// One thread per element over @p n elements.
  explicit launch_cfg(std::int64_t n, std::source_location loc = std::source_location::current())
      : grid(narrow<unsigned>(ceil_div(n, kBlock1d))), block(kBlock1d), loc(loc)
  {
  }

  /// One thread per pixel over an @p nrow x @p ncol plane: x runs over columns, y over rows.
  explicit launch_cfg(std::int64_t nrow, std::int64_t ncol, std::source_location loc = std::source_location::current())
      : grid(narrow<unsigned>(ceil_div(ncol, kBlock2d.x)), narrow<unsigned>(ceil_div(nrow, kBlock2d.y))),
        block(kBlock2d),
        loc(loc)
  {
  }

  /// One thread per pixel over @p depth planes of @p nrow x @p ncol: z runs over planes, at most 65535.
  explicit launch_cfg(std::int64_t depth, std::int64_t nrow, std::int64_t ncol,
                      std::source_location loc = std::source_location::current())
      : grid(narrow<unsigned>(ceil_div(ncol, kBlock2d.x)), narrow<unsigned>(ceil_div(nrow, kBlock2d.y)),
             narrow<unsigned>(depth)),
        block(kBlock2d),
        loc(loc)
  {
  }

  /**
   * @brief Any other shape, given as an explicit @p grid and @p block.
   *
   * A named factory, not a constructor: launch_cfg(1, 1) would silently pick the 2D overload.
   */
  static launch_cfg custom(dim3 grid, dim3 block, std::source_location loc = std::source_location::current())
  {
    launch_cfg cfg;
    cfg.grid = grid;
    cfg.block = block;
    cfg.loc = loc;
    return cfg;
  }

 private:
  launch_cfg() = default;
};

/**
 * @brief Launch a kernel on the context's stream and check the launch.
 *
 * An empty grid is a no-op: CUDA rejects it as an invalid configuration.
 *
 * @param[in] ctx     Context whose stream runs the kernel.
 * @param[in] cfg     Grid, block and call site.
 * @param[in] kernel  The __global__ function.
 * @param[in] args    Kernel arguments, converted to its parameter types.
 * @throws std::runtime_error If the launch fails, naming the call site recorded in @p cfg.
 */
template <typename... KArgs, typename... Args>
void launch(const exec_ctx& ctx, launch_cfg cfg, void (*kernel)(KArgs...), Args&&... args)
{
  if (cfg.grid.x == 0 || cfg.grid.y == 0 || cfg.grid.z == 0) return;
  kernel<<<cfg.grid, cfg.block, 0, ctx.cuda_stream>>>(std::forward<Args>(args)...);
  if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess)
    throw std::runtime_error(std::string("CUDA launch error at ") + cfg.loc.file_name() + ":" +
                             std::to_string(cfg.loc.line()) + ": " + cudaGetErrorString(err));
}

}  // namespace fast_deconv::core
