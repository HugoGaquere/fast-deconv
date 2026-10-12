#pragma once

#include <gtest/gtest.h>

#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::test {

// Base fixture for every backend test: provides per-test core::exec_resources on
// device 0. A backend that cannot supply one fails the test rather than skipping
// it. Streams are deliberately not cached here — core::exec_ctx is non-movable,
// so tests create theirs locally with `const auto sr = res().make_ctx();`.
class BackendTest : public ::testing::Test {
 protected:
  core::exec_resources& res() { return res_; }

 private:
  core::exec_resources res_{0};
};

}  // namespace fast_deconv::test
