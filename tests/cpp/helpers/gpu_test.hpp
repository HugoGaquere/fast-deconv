#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <fast_deconv/core/exec_ctx.hpp>
#include <optional>

// Skip the current test when no CUDA device is usable, instead of crashing at
// the first CUDA call. Usable from any test body or SetUp().
#define SKIP_IF_NO_GPU()                                                                         \
  do {                                                                                           \
    int fd_test_device_count_ = 0;                                                               \
    if (cudaGetDeviceCount(&fd_test_device_count_) != cudaSuccess || fd_test_device_count_ == 0) \
      GTEST_SKIP() << "No CUDA device available";                                                \
  } while (0)

namespace fast_deconv::test {

// Base fixture for every GPU test: skips cleanly on machines without a CUDA
// device, then provides per-test core::exec_resources on device 0. Streams are
// deliberately not cached here — core::exec_ctx is non-movable, so tests create
// theirs locally with `const auto sr = res().make_ctx();`.
class GpuTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    SKIP_IF_NO_GPU();
    res_.emplace(0);
  }

  core::exec_resources& res() { return *res_; }

 private:
  // core::exec_resources is non-movable: construct in place once the GPU check passed.
  std::optional<core::exec_resources> res_;
};

}  // namespace fast_deconv::test
