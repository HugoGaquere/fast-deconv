#include <gtest/gtest.h>

#include <cstdint>
#include <fast_deconv/common/region.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/util/utils.hpp>
#include <stdexcept>
#include <type_traits>

namespace core = fast_deconv::core;
namespace common = fast_deconv::common;

namespace {
constexpr std::int64_t kInt32Max = 2147483647;
}  // namespace

// Cubes past INT_MAX elements (10 scales over a 19845^2 plane, 6 x 20000^2 dirty) segfaulted when offsets
// wrapped in int32. Every span's extents come from core::dims, so pinning its index type covers them all.
static_assert(std::is_same_v<core::span3d<float>::index_type, std::int64_t>);
static_assert(std::is_same_v<core::host_span4d<float>::index_type, std::int64_t>);

// The clean loop's peak unravel and footprint clipping, on the corner of a 20000^2 plane.
TEST(LargeIndex, CornerPeakOfLargePlane)
{
  constexpr std::int64_t kSide = 20000;

  const common::index2d peak = fast_deconv::util::unravel_index_2D(kSide * kSide - 1, kSide);
  EXPECT_EQ(peak.row, kSide - 1);
  EXPECT_EQ(peak.col, kSide - 1);

  // A 33 x 33 footprint centered on the corner keeps its top-left quarter, ending on the last pixel.
  const auto ovr = common::compute_overlap_region(peak, kSide, kSide, 33, 33);
  EXPECT_EQ(ovr.nrow, 17);
  EXPECT_EQ(ovr.ncol, 17);
  EXPECT_EQ((ovr.arow0 + ovr.nrow - 1) * ovr.lda + ovr.acol0 + ovr.ncol - 1, kSide * kSide - 1);
}

TEST(LargeIndex, NarrowThrowsOnlyWhenTheValueDoesNotFit)
{
  EXPECT_EQ(core::narrow<int>(kInt32Max), kInt32Max);
  EXPECT_THROW(core::narrow<int>(kInt32Max + 1), std::overflow_error);
  EXPECT_THROW(core::narrow<unsigned>(std::int64_t{-1}), std::overflow_error);
  EXPECT_THROW(core::narrow<std::size_t>(std::int64_t{-1}), std::overflow_error);
}
