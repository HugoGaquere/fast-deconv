#include <gtest/gtest.h>

#include <cstdint>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/util/utils.hpp>
#include <utility>
#include <vector>

// Pure index math and host-side bookkeeping — no GPU work. The device spans
// below are built on a fake pointer that is never dereferenced; slice_leading
// is plain pointer/extent arithmetic.

namespace core = fast_deconv::core;
namespace util = fast_deconv::util;
namespace wscms = fast_deconv::algorithm::wscms;

// ============================================================================
// util::unravel_index_2D — row-major flat index → (y, x)
// ============================================================================

TEST(UnravelIndex2D, CornersAndInteriorOnNonSquareGrid)
{
  const int width = 7;  // grid is (height=5, width=7), deliberately non-square

  EXPECT_EQ(util::unravel_index_2D(0, width), (std::pair<int, int>{0, 0}));
  EXPECT_EQ(util::unravel_index_2D(width - 1, width), (std::pair<int, int>{0, width - 1}));
  EXPECT_EQ(util::unravel_index_2D(width, width), (std::pair<int, int>{1, 0}));

  const int y = 3, x = 4;
  EXPECT_EQ(util::unravel_index_2D(y * width + x, width), (std::pair<int, int>{y, x}));

  const int last = 5 * width - 1;
  EXPECT_EQ(util::unravel_index_2D(last, width), (std::pair<int, int>{4, width - 1}));
}

// ============================================================================
// core::slice_leading — leading-dimension slices preserve trailing extents and
// advance the data pointer by i * stride(0).
// ============================================================================

namespace {
float* fake_base() { return reinterpret_cast<float*>(0x1000); }
}  // namespace

TEST(SliceLeading, Span3dToSpan2d)
{
  core::device_span3d<float> s3(fake_base(), 4, 5, 6);
  auto s2 = core::slice_leading(s3, 2);
  EXPECT_EQ(s2.data_handle(), fake_base() + 2 * 5 * 6);
  EXPECT_EQ(s2.extent(0), 5);
  EXPECT_EQ(s2.extent(1), 6);
}

TEST(SliceLeading, Span4dToSpan3d)
{
  core::device_span4d<float> s4(fake_base(), 3, 4, 5, 6);
  auto s3 = core::slice_leading(s4, 1);
  EXPECT_EQ(s3.data_handle(), fake_base() + 1 * 4 * 5 * 6);
  EXPECT_EQ(s3.extent(0), 4);
  EXPECT_EQ(s3.extent(1), 5);
  EXPECT_EQ(s3.extent(2), 6);
}

TEST(SliceLeading, Span5dToSpan4d)
{
  core::device_span5d<float> s5(fake_base(), 2, 3, 4, 5, 6);
  auto s4 = core::slice_leading(s5, 1);
  EXPECT_EQ(s4.data_handle(), fake_base() + 1 * 3 * 4 * 5 * 6);
  EXPECT_EQ(s4.extent(0), 3);
  EXPECT_EQ(s4.extent(1), 4);
  EXPECT_EQ(s4.extent(2), 5);
  EXPECT_EQ(s4.extent(3), 6);
}

TEST(SliceLeading, IndexZeroIsIdentityView)
{
  core::device_span3d<float> s3(fake_base(), 4, 5, 6);
  auto s2 = core::slice_leading(s3, 0);
  EXPECT_EQ(s2.data_handle(), fake_base());
  EXPECT_EQ(s2.extent(0), 5);
  EXPECT_EQ(s2.extent(1), 6);
}

// ============================================================================
// wscms_result::add_component — parallel-array bookkeeping
// ============================================================================

TEST(WscmsResult, AddComponentKeepsParallelArraysInSync)
{
  wscms::wscms_result result(/*max_iter=*/10, /*coeff_order=*/2);
  EXPECT_TRUE(result.peak_coords.empty());
  EXPECT_TRUE(result.scales.empty());
  EXPECT_TRUE(result.gains.empty());

  result.add_component({3, 4}, /*scale=*/1, /*gain=*/0.1f);
  result.add_component({7, 2}, /*scale=*/0, /*gain=*/0.2f);

  ASSERT_EQ(result.peak_coords.size(), 2u);
  ASSERT_EQ(result.scales.size(), 2u);
  ASSERT_EQ(result.gains.size(), 2u);

  EXPECT_EQ(result.peak_coords.at(0), (std::pair<int, int>{3, 4}));
  EXPECT_EQ(result.peak_coords.at(1), (std::pair<int, int>{7, 2}));
  EXPECT_EQ(result.scales.at(0), 1);
  EXPECT_EQ(result.scales.at(1), 0);
  EXPECT_FLOAT_EQ(result.gains.at(0), 0.1f);
  EXPECT_FLOAT_EQ(result.gains.at(1), 0.2f);
}
