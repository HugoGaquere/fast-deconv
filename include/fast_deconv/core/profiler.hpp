#pragma once

// NVTX (CUDA) / Tracy (host) front end, enabled by FAST_DECONV_WITH_PROFILER.
//   FD_PROFILE_FN();                        // zone named after the enclosing function
//   FD_PROFILE_SCOPE("clean_loop");         // zone, literal name
//   FD_PROFILE_SCOPE_FMT("outer[{}]", it);  // zone, fmt-formatted name
//   FD_PROFILE_MARK("checkpoint");          // instantaneous event, literal only
// Tracy only:
//   FD_PROFILE_FRAME();  FD_PROFILE_PLOT("rms", value);  FD_PROFILE_APPINFO(text);
//   FD_PROFILE_ALLOC(ptr, n);  FD_PROFILE_FREE(ptr);

#if defined(TRACY_ENABLE)

#include <fmt/format.h>

#include <string>
#include <tracy/Tracy.hpp>

#define FD_PROFILE_FN() ZoneScoped
#define FD_PROFILE_SCOPE(name) ZoneScopedN(name)
#define FD_PROFILE_SCOPE_FMT(...)                                    \
  ZoneScoped;                                                        \
  const ::std::string _fd_profile_label{::fmt::format(__VA_ARGS__)}; \
  ZoneName(_fd_profile_label.data(), _fd_profile_label.size())
#define FD_PROFILE_MARK(msg) TracyMessageL(msg)
#define FD_PROFILE_FRAME() FrameMark
// Overloaded on int64_t/float/double, so callers cast ints.
#define FD_PROFILE_PLOT(name, value) TracyPlot(name, value)
// Bound once: TracyAppInfo reads its argument twice.
#define FD_PROFILE_APPINFO(text)                                    \
  do {                                                              \
    const ::std::string _fd_profile_info{text};                     \
    TracyAppInfo(_fd_profile_info.data(), _fd_profile_info.size()); \
  } while (0)
#define FD_PROFILE_ALLOC(ptr, num_bytes) TracyAlloc(ptr, num_bytes)
#define FD_PROFILE_FREE(ptr) TracyFree(ptr)

#elif defined(FD_NVTX_ENABLE)

#include <fmt/format.h>

#include <nvtx3/nvtx3.hpp>
#include <string>

namespace fast_deconv::profiler {

/// Dedicated "fast_deconv" track in Nsight Systems.
struct domain {
  static constexpr char const* name{"fast_deconv"};
};

}  // namespace fast_deconv::profiler

#define FD_PROFILE_NVTX_RANGE(name) \
  ::nvtx3::scoped_range_in<::fast_deconv::profiler::domain> _fd_profile_zone { name }
#define FD_PROFILE_FN() FD_PROFILE_NVTX_RANGE(__func__)
#define FD_PROFILE_SCOPE(name) FD_PROFILE_NVTX_RANGE(name)
// Label declared first so it outlives the range.
#define FD_PROFILE_SCOPE_FMT(...)                                    \
  const ::std::string _fd_profile_label{::fmt::format(__VA_ARGS__)}; \
  FD_PROFILE_NVTX_RANGE(_fd_profile_label.c_str())
#define FD_PROFILE_MARK(msg) ::nvtx3::mark_in<::fast_deconv::profiler::domain>(msg)

#else

#define FD_PROFILE_FN() ((void)0)
#define FD_PROFILE_SCOPE(name) ((void)0)
#define FD_PROFILE_SCOPE_FMT(...) ((void)0)
#define FD_PROFILE_MARK(msg) ((void)0)

#endif

#if !defined(TRACY_ENABLE)
#define FD_PROFILE_FRAME() ((void)0)
#define FD_PROFILE_PLOT(name, value) ((void)0)
#define FD_PROFILE_APPINFO(text) ((void)0)
#define FD_PROFILE_ALLOC(ptr, num_bytes) ((void)0)
#define FD_PROFILE_FREE(ptr) ((void)0)
#endif
