#include <gtest/gtest.h>

#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>

#include "helpers/backend_test.hpp"

namespace core = fast_deconv::core;
namespace fdtest = fast_deconv::test;

namespace {

constexpr int kRows = 37;
constexpr int kCols = 53;

}  // namespace

class Resources : public fdtest::BackendTest {};

TEST_F(Resources, MdcontainerAllocFreeRoundTrip)
{
  const auto sr = res().make_ctx();
  const uint64_t baseline = res().pool_used_bytes();

  {
    auto buf = sr.alloc_mdcontainer_async<float>(kRows, kCols);
    ASSERT_NE(buf.data_handle(), nullptr);
    EXPECT_EQ(buf.extent(0), kRows);
    EXPECT_EQ(buf.extent(1), kCols);
    EXPECT_EQ(buf.use_count(), 1);
    sr.wait();
    EXPECT_GE(res().pool_used_bytes(), baseline + kRows * kCols * sizeof(float));
  }

  sr.wait();
  EXPECT_EQ(res().pool_used_bytes(), baseline);
}

TEST_F(Resources, PoolHandsBackAFreedBlock)
{
  const auto sr = res().make_ctx();

  void* first = nullptr;
  {
    auto buf = sr.alloc_mdcontainer_async<float>(kRows, kCols);
    first = buf.data_handle();
  }
  sr.wait();

  const auto again = sr.alloc_mdcontainer_async<float>(kRows, kCols);
  EXPECT_EQ(again.data_handle(), first);
}
