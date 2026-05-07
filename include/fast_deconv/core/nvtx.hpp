#pragma once

// Wrap NVTX3 ranges/marks behind project macros so the rest of the codebase
// never includes NVTX directly (mirrors how logger.hpp wraps spdlog).
//
// Probes are enabled by default. To compile them out at zero cost, build with:
//   -DFD_NVTX_DISABLE
//
// Usage:
//   FD_NVTX_RANGE("init");                       // static name
//   FD_NVTX_RANGE("outer_iter[{}]", iter_idx);   // fmt-formatted dynamic name
//   FD_NVTX_RANGE_FN();                          // range named after enclosing fn
//   FD_NVTX_MARK("checkpoint reached");          // instantaneous event

#if defined(FD_NVTX_DISABLE)

#define FD_NVTX_RANGE(...) ((void)0)
#define FD_NVTX_RANGE_FN() ((void)0)
#define FD_NVTX_MARK(...) ((void)0)

#else

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

#include <string>
#include <utility>

namespace fast_deconv::nvtx {

/// All ranges/marks emitted via this wrapper land in the "fast_deconv" NVTX
/// domain, so they show up as a dedicated track in Nsight Systems.
struct domain {
  static constexpr char const* name{"fast_deconv"};
};

/// RAII range that owns its label. The owned `std::string` keeps dynamic
/// names (fmt-formatted, std::string&&, etc.) alive until the inner
/// nvtx3 range pops at scope exit.
class scoped_range {
  std::string label_;
  ::nvtx3::scoped_range_in<domain> range_;

 public:
  explicit scoped_range(const char* name) : label_{name}, range_{label_.c_str()} {}
  explicit scoped_range(std::string name) : label_{std::move(name)}, range_{label_.c_str()} {}

  template <typename... Args>
  scoped_range(::fmt::format_string<Args...> fmt_str, Args&&... args)
      : label_{::fmt::format(fmt_str, std::forward<Args>(args)...)}, range_{label_.c_str()}
  {
  }

  scoped_range(const scoped_range&) = delete;
  scoped_range& operator=(const scoped_range&) = delete;
};

inline void mark(const char* msg) { ::nvtx3::mark_in<domain>(msg); }
inline void mark(const std::string& msg) { ::nvtx3::mark_in<domain>(msg.c_str()); }

}  // namespace fast_deconv::nvtx

#define FD_NVTX_PASTE_INNER(a, b) a##b
#define FD_NVTX_PASTE(a, b) FD_NVTX_PASTE_INNER(a, b)

#define FD_NVTX_RANGE(...) \
  ::fast_deconv::nvtx::scoped_range FD_NVTX_PASTE(_fd_nvtx_range_, __LINE__) { __VA_ARGS__ }

#define FD_NVTX_RANGE_FN() FD_NVTX_RANGE(__func__)

#define FD_NVTX_MARK(...) ::fast_deconv::nvtx::mark(__VA_ARGS__)

#endif
