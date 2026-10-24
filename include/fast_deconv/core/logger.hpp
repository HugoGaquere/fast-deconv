#pragma once

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <fast_deconv/core/concepts.hpp>

// fmt::formatter for any mdspan type (host or device).
// Formats the shape as [d0xd1x...], e.g. [4x256x256x2].
template <typename M>
  requires fast_deconv::core::cpts::mdspan<M>
struct fmt::formatter<M> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  auto format(const M& m, fmt::format_context& ctx) const
  {
    auto it = fmt::format_to(ctx.out(), "[");
    for (std::size_t i = 0; i < M::rank(); ++i) {
      if (i > 0) it = fmt::format_to(it, ",");
      it = fmt::format_to(it, "{}", m.extent(i));
    }
    return fmt::format_to(it, "]");
  }
};

// Wrap spdlog macros so the rest of the codebase never includes spdlog directly.
// These use SPDLOG_LOGGER_* macros which are compiled out at preprocessing time
// when SPDLOG_ACTIVE_LEVEL is set higher than the call's level.
//
// To disable all logging at compile time (zero overhead), build with:
//   -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF
//
// Levels: TRACE=0, DEBUG=1, INFO=2, WARN=3, ERROR=4, CRITICAL=5, OFF=6
// Default is INFO (trace and debug compiled out).

#define FD_LOG_LEVEL_TRACE SPDLOG_LEVEL_TRACE
#define FD_LOG_LEVEL_DEBUG SPDLOG_LEVEL_DEBUG
#define FD_LOG_LEVEL_INFO SPDLOG_LEVEL_INFO
#define FD_LOG_LEVEL_WARN SPDLOG_LEVEL_WARN
#define FD_LOG_LEVEL_ERROR SPDLOG_LEVEL_ERROR
#define FD_LOG_LEVEL_CRITICAL SPDLOG_LEVEL_CRITICAL
#define FD_LOG_LEVEL_OFF SPDLOG_LEVEL_OFF
#define FD_LOG_ACTIVE_LEVEL SPDLOG_ACTIVE_LEVEL

// Trace and debug lines carry [file:line]; info and above pass an empty
// source_loc so the default pattern omits it.
#define FD_LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define FD_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)

#define FD_LOG_NO_LOC_(lvl, ...) spdlog::default_logger_raw()->log(spdlog::source_loc{}, lvl, __VA_ARGS__)

#if FD_LOG_ACTIVE_LEVEL <= FD_LOG_LEVEL_INFO
#define FD_LOG_INFO(...) FD_LOG_NO_LOC_(spdlog::level::info, __VA_ARGS__)
#else
#define FD_LOG_INFO(...) (void)0
#endif

#if FD_LOG_ACTIVE_LEVEL <= FD_LOG_LEVEL_WARN
#define FD_LOG_WARN(...) FD_LOG_NO_LOC_(spdlog::level::warn, __VA_ARGS__)
#else
#define FD_LOG_WARN(...) (void)0
#endif

#if FD_LOG_ACTIVE_LEVEL <= FD_LOG_LEVEL_ERROR
#define FD_LOG_ERROR(...) FD_LOG_NO_LOC_(spdlog::level::err, __VA_ARGS__)
#else
#define FD_LOG_ERROR(...) (void)0
#endif

#if FD_LOG_ACTIVE_LEVEL <= FD_LOG_LEVEL_CRITICAL
#define FD_LOG_CRITICAL(...) FD_LOG_NO_LOC_(spdlog::level::critical, __VA_ARGS__)
#else
#define FD_LOG_CRITICAL(...) (void)0
#endif

namespace fast_deconv::log {

inline void set_level(spdlog::level::level_enum level) { spdlog::set_level(level); }

}  // namespace fast_deconv::log
