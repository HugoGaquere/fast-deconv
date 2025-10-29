#pragma once

#include <fast_deconv/core/stream_resources.hpp>

namespace fast_deconv::core {

inline fast_deconv::core::stream_resources& cached_stream_resources()
{
  thread_local fast_deconv::core::stream_resources resources{};
  return resources;
}

}  // namespace fast_deconv::python::detail
