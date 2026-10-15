#include <gtest/gtest.h>

#include <cstdint>
#include <fast_deconv/common/region.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>
#include <fast_deconv/util/utils.hpp>
#include <stdexcept>

namespace core = fast_deconv::core;
namespace common = fast_deconv::common;
namespace linalg = fast_deconv::linalg;

namespace {
constexpr std::int64_t kInt32Max = 2147483647;
// Mappings are pure arithmetic, so these views never allocate or dereference.
float* const kNoData = nullptr;
}  // namespace

// The replay case that segfaulted: 10 scales over a 19845^2 plane is 3.9e9
// elements, so every offset past scale 5 wraps if the index type is int32.
TEST(LargeIndex, ScaleCubeOffsetsExceedInt32)
{
  const std::int64_t plane = std::int64_t{19845} * 19845;
  ASSERT_GT(10 * plane, kInt32Max);

  core::host_span3d<float> cube(kNoData, 10, 19845, 19845);
  EXPECT_EQ(cube.mapping()(9, 0, 0), 9 * plane);
  EXPECT_EQ(cube.mapping().required_span_size(), 10 * plane);
}

// Same for the (n_facets, n_freq, psf, psf) stack psf_convolution slices per facet.
TEST(LargeIndex, PsfStackOffsetsExceedInt32)
{
  const std::int64_t per_facet = std::int64_t{8} * 1721 * 1721;
  ASSERT_GT(121 * per_facet, kInt32Max);

  core::host_span4d<float> psfs(kNoData, 121, 8, 1721, 1721);
  EXPECT_EQ(psfs.mapping()(120, 0, 0, 0), 120 * per_facet);
}

// The large test's 6 x 20000^2 dirty cube: the clean loop's offset to the last element of the last plane,
// built from the same helpers it uses (unravel, overlap region, plane stride), lands past INT_MAX exactly.
TEST(LargeIndex, DirtyCubeLastElementOffset)
{
  constexpr std::int64_t kFreq = 6;
  constexpr std::int64_t kSide = 20000;
  constexpr std::int64_t kPlane = kSide * kSide;

  core::host_span3d<float> cube(kNoData, kFreq, kSide, kSide);
  const std::int64_t last = cube.mapping()(kFreq - 1, kSide - 1, kSide - 1);
  ASSERT_GT(last, kInt32Max);
  EXPECT_EQ(last, kFreq * kPlane - 1);
  EXPECT_EQ(cube.mapping().stride(0), kPlane);
  EXPECT_EQ(cube.mapping().required_span_size(), kFreq * kPlane);

  const common::index2d peak = fast_deconv::util::unravel_index_2D(kPlane - 1, kSide);
  EXPECT_EQ(peak.row, kSide - 1);
  EXPECT_EQ(peak.col, kSide - 1);

  // A 33 x 33 footprint centered on the corner keeps its top-left quarter, ending on the last pixel.
  const auto ovr = common::compute_overlap_region(peak, kSide, kSide, 33, 33);
  EXPECT_EQ(ovr.nrow, 17);
  EXPECT_EQ(ovr.ncol, 17);
  const std::int64_t last_in_footprint =
      (kFreq - 1) * cube.mapping().stride(0) + (ovr.arow0 + ovr.nrow - 1) * ovr.lda + ovr.acol0 + ovr.ncol - 1;
  EXPECT_EQ(last_in_footprint, last);
}

TEST(LargeIndex, NarrowThrowsOnlyWhenTheValueDoesNotFit)
{
  EXPECT_EQ(core::narrow<int>(kInt32Max), kInt32Max);
  EXPECT_THROW(core::narrow<int>(kInt32Max + 1), std::overflow_error);
  EXPECT_THROW(core::narrow<unsigned>(std::int64_t{-1}), std::overflow_error);
  EXPECT_THROW(core::narrow<std::size_t>(std::int64_t{-1}), std::overflow_error);
}

// The padded plane is no longer capped at INT_MAX: cuFFT gets it through its 64-bit plan.
TEST(LargeIndex, PaddedFftPlaneIsNotCapped)
{
  const linalg::fft_dims dims(19845, 19845, /*gap=*/30155);  // padded 50000^2
  EXPECT_GT(dims.padded_total(), kInt32Max);
}
