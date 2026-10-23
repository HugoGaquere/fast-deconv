#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <fast_deconv/core/dims.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <iostream>
#include <source_location>
#include <utility>

namespace fast_deconv::core {

// Default block shapes: 1D kernels take one thread per element, 2D kernels one per pixel. 32 wide so a warp reads one
// contiguous row segment.
inline constexpr unsigned kBlock1d = 256;
inline constexpr dim3 kBlock2d(32, 8);

// Launch shape plus the caller's location, so a failed launch reports the call site, not this header.
struct launch_cfg {
  dim3 grid;
  dim3 block;
  std::source_location loc;

  // One thread per element over n elements.
  explicit launch_cfg(std::int64_t n, std::source_location loc = std::source_location::current())
      : grid(narrow<unsigned>(ceil_div(n, kBlock1d))), block(kBlock1d), loc(loc)
  {
  }

  // One thread per pixel over an nrow x ncol plane: x runs over columns, y over rows.
  explicit launch_cfg(std::int64_t nrow, std::int64_t ncol, std::source_location loc = std::source_location::current())
      : grid(narrow<unsigned>(ceil_div(ncol, kBlock2d.x)), narrow<unsigned>(ceil_div(nrow, kBlock2d.y))),
        block(kBlock2d),
        loc(loc)
  {
  }

  // Any other shape. A named factory, not a constructor: launch_cfg(1, 1) would silently pick the 2D overload.
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

// Launch on the context's stream and check the launch. An empty grid is a no-op: CUDA rejects it as an invalid
// configuration.
template <typename... KArgs, typename... Args>
void launch(const exec_ctx& ctx, launch_cfg cfg, void (*kernel)(KArgs...), Args&&... args)
{
  if (cfg.grid.x == 0 || cfg.grid.y == 0 || cfg.grid.z == 0) return;
  kernel<<<cfg.grid, cfg.block, 0, ctx.cuda_stream>>>(std::forward<Args>(args)...);
  if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
    std::cerr << "CUDA launch error at: " << cfg.loc.file_name() << ":" << cfg.loc.line() << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace fast_deconv::core
