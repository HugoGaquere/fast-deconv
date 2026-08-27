#pragma once

#include <cstddef>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/matrix/peak.hpp>

namespace fast_deconv::matrix {

/// Host argmax is a linear scan, so there is no temp storage to hoist. The
/// type exists to keep the call sites identical across backends.
class argmax_ctx {
 public:
  argmax_ctx(const core::exec_ctx& ctx, std::size_t n_elements) : ctx_(ctx), n_elements_(n_elements) {}

  argmax_ctx(const argmax_ctx&) = delete;
  argmax_ctx& operator=(const argmax_ctx&) = delete;

  void run_async(core::span2d<float> data);
  peak run(core::span2d<float> data);

  std::size_t size() const { return n_elements_; }

 private:
  const core::exec_ctx& ctx_;
  std::size_t n_elements_;
  peak last_{};
};

}  // namespace fast_deconv::matrix
