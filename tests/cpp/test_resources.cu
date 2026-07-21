#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <optional>

#include "helpers/gpu_test.hpp"

namespace core = fast_deconv::core;
namespace fdtest = fast_deconv::test;

namespace {

constexpr int kRows = 37;
constexpr int kCols = 53;

core::device_span2d<float> pass_by_value(core::device_span2d<float> s) { return s; }

}  // namespace

class Resources : public fdtest::GpuTest {};

TEST_F(Resources, MdcontainerAllocFreeRoundTrip)
{
  const auto sr = res().make_stream();
  const uint64_t baseline = res().pool_used_bytes();

  {
    auto buf = sr.alloc_mdcontainer_async<float>(kRows, kCols);
    ASSERT_NE(buf.data_handle(), nullptr);
    EXPECT_EQ(buf.extent(0), kRows);
    EXPECT_EQ(buf.extent(1), kCols);
    EXPECT_EQ(buf.use_count(), 1);
    sr.sync();
    EXPECT_GE(res().pool_used_bytes(), baseline + kRows * kCols * sizeof(float));
  }

  sr.sync();
  EXPECT_EQ(res().pool_used_bytes(), baseline);
}

TEST_F(Resources, MdcontainerCopySharesOwnership)
{
  const auto sr = res().make_stream();
  const uint64_t baseline = res().pool_used_bytes();

  std::optional original{sr.alloc_mdcontainer_async<float>(kRows, kCols)};
  float* const ptr = original->data_handle();
  core::device_cont2d<float> copy = *original;
  EXPECT_EQ(copy.use_count(), 2);

  // The copy must keep the allocation alive past the original's destruction.
  original.reset();
  sr.sync();
  EXPECT_EQ(copy.data_handle(), ptr);
  EXPECT_EQ(copy.use_count(), 1);
  EXPECT_GE(res().pool_used_bytes(), baseline + kRows * kCols * sizeof(float));

  copy = core::device_cont2d<float>{};
  sr.sync();
  EXPECT_EQ(res().pool_used_bytes(), baseline);
}

TEST_F(Resources, MdcontainerDecaysToSpan)
{
  const auto sr = res().make_stream();
  auto buf = sr.alloc_mdcontainer_async<float>(kRows, kCols);

  const core::device_span2d<float> view = pass_by_value(buf);
  EXPECT_EQ(view.data_handle(), buf.data_handle());
  EXPECT_EQ(view.extent(0), kRows);
  EXPECT_EQ(view.extent(1), kCols);
  // Decay builds a plain view: no ownership share, refcount untouched.
  EXPECT_EQ(buf.use_count(), 1);
  sr.sync();
}

TEST_F(Resources, MdcontainerDefaultConstructedIsEmpty)
{
  const core::device_cont3d<bool> empty;
  EXPECT_EQ(empty.data_handle(), nullptr);
  EXPECT_EQ(empty.use_count(), 0);
}
