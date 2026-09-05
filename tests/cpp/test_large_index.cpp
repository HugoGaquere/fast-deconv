#include <gtest/gtest.h>

#include <cstdint>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft_dims.hpp>
#include <stdexcept>

namespace core = fast_deconv::core;
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

// Same for the (n_facets, n_freq, psf, psf) stack conv_psf_cache slices per facet.
TEST(LargeIndex, PsfStackOffsetsExceedInt32)
{
  const std::int64_t per_facet = std::int64_t{8} * 1721 * 1721;
  ASSERT_GT(121 * per_facet, kInt32Max);

  core::host_span4d<float> psfs(kNoData, 121, 8, 1721, 1721);
  EXPECT_EQ(psfs.mapping()(120, 0, 0, 0), 120 * per_facet);
}

// A single plane must still fit an int: the kernels hold a plane offset in one.
TEST(LargeIndex, PaddedFftPlaneIsGuarded)
{
  EXPECT_NO_THROW(linalg::fft_dims(19845, 19845, 1.1f));                      // padded 21870^2, fits
  EXPECT_THROW(linalg::fft_dims(19845, 19845, 2.5f), std::invalid_argument);  // 50000^2, does not
}
