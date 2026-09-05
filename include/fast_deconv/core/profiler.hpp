#pragma once

// Single instrumentation front end for the project: these macros fan out to
// NVTX (Nsight Systems, next to the GPU timeline) and to Tracy (host timeline).
// Nothing else in the codebase includes either tool directly (mirrors how
// logger.hpp wraps spdlog).
//
// NVTX is on unless the build defines FD_NVTX_DISABLE.
// Tracy is on when the Tracy client is linked (CMake option
// FAST_DECONV_WITH_TRACY, or ./scripts/build.sh --tracy), which defines TRACY_ENABLE.
//
// Usage:
//   FD_PROFILE_FN();                        // zone named after the enclosing function
//   FD_PROFILE_SCOPE("clean_loop");         // zone, literal name
//   FD_PROFILE_SCOPE_FMT("outer[{}]", it);  // zone, fmt-formatted name
//   FD_PROFILE_MARK("checkpoint");          // instantaneous event
//   FD_PROFILE_FRAME();                     // end-of-frame marker (Tracy only)
//
// One zone per scope: Tracy names its RAII object after the scope, so two zones
// in the same scope collide. Open a nested block if you need both.

//-------------------------------------------------------------------//
// Tracy
//-------------------------------------------------------------------//
#if defined(TRACY_ENABLE)

#define FD_PROFILE_HAS_TRACY 1

#include <cstring>
#include <tracy/Tracy.hpp>

#define FD_PROFILE_TRACY_FN() ZoneScoped
#define FD_PROFILE_TRACY_SCOPE(name) ZoneScopedN(name)
// Tracy copies the text into its queue, so the label needs no lifetime care.
#define FD_PROFILE_TRACY_NAME(label) ZoneName((label).data(), (label).size())
#define FD_PROFILE_TRACY_MARK(msg) TracyMessage(msg, ::std::strlen(msg))
#define FD_PROFILE_FRAME() FrameMark

#else

#define FD_PROFILE_TRACY_FN() ((void)0)
#define FD_PROFILE_TRACY_SCOPE(name) ((void)0)
#define FD_PROFILE_TRACY_NAME(label) ((void)0)
#define FD_PROFILE_TRACY_MARK(msg) ((void)0)
#define FD_PROFILE_FRAME() ((void)0)

#endif

//-------------------------------------------------------------------//
// NVTX
//-------------------------------------------------------------------//
#if !defined(FD_NVTX_DISABLE)

#define FD_PROFILE_HAS_NVTX 1

#include <nvtx3/nvtx3.hpp>

namespace fast_deconv::profiler {

/// All ranges/marks emitted here land in the "fast_deconv" NVTX domain, so they
/// show up as a dedicated track in Nsight Systems.
struct domain {
  static constexpr char const* name{"fast_deconv"};
};

using scoped_range = ::nvtx3::scoped_range_in<domain>;

inline void mark(const char* msg) { ::nvtx3::mark_in<domain>(msg); }

}  // namespace fast_deconv::profiler

#define FD_PROFILE_NVTX_RANGE(name) \
  ::fast_deconv::profiler::scoped_range _fd_profile_zone { name }
#define FD_PROFILE_NVTX_MARK(msg) ::fast_deconv::profiler::mark(msg)

#else

#define FD_PROFILE_NVTX_RANGE(name) ((void)0)
#define FD_PROFILE_NVTX_MARK(msg) ((void)0)

#endif

//-------------------------------------------------------------------//
// Common API
//-------------------------------------------------------------------//
#if defined(FD_PROFILE_HAS_TRACY) || defined(FD_PROFILE_HAS_NVTX)

#include <fmt/format.h>

#include <string>

#define FD_PROFILE_FN()  \
  FD_PROFILE_TRACY_FN(); \
  FD_PROFILE_NVTX_RANGE(__func__)

#define FD_PROFILE_SCOPE(name)  \
  FD_PROFILE_TRACY_SCOPE(name); \
  FD_PROFILE_NVTX_RANGE(name)

// The label outlives both probes: it is declared before them, so it is
// destroyed after them.
#define FD_PROFILE_SCOPE_FMT(...)                                    \
  FD_PROFILE_TRACY_FN();                                             \
  const ::std::string _fd_profile_label{::fmt::format(__VA_ARGS__)}; \
  FD_PROFILE_TRACY_NAME(_fd_profile_label);                          \
  FD_PROFILE_NVTX_RANGE(_fd_profile_label.c_str())

#define FD_PROFILE_MARK(msg)    \
  do {                          \
    FD_PROFILE_TRACY_MARK(msg); \
    FD_PROFILE_NVTX_MARK(msg);  \
  } while (0)

#else

#define FD_PROFILE_FN() ((void)0)
#define FD_PROFILE_SCOPE(name) ((void)0)
#define FD_PROFILE_SCOPE_FMT(...) ((void)0)
#define FD_PROFILE_MARK(msg) ((void)0)

#endif
