#pragma once

#include <fmt/format.h>
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

#define FD_LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define FD_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define FD_LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define FD_LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define FD_LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define FD_LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)

namespace fast_deconv::log {

inline void set_level(spdlog::level::level_enum level) { spdlog::set_level(level); }

}  // namespace fast_deconv::log
