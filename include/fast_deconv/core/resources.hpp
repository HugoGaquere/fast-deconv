#pragma once
#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::core {

// Legacy spellings kept so existing call sites keep building while they migrate
// to exec_ctx / exec_resources. Delete once nothing names them.
using stream_resources = exec_ctx;
using resources = exec_resources;

template <typename T>
using device_ptr = owned_ptr<T>;

}  // namespace fast_deconv::core
