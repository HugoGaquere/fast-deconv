#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <climits>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/morphology/dilation.hpp>
#include <fast_deconv/morphology/roi.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <vector>

#include "helpers/device_buffers.hpp"
#include "helpers/gpu_test.hpp"

namespace core = fast_deconv::core;
namespace morpho = fast_deconv::morphology;
namespace fdtest = fast_deconv::test;

// ============================================================================
// compute_mask_roi
//
// Property-based oracle tests on small non-square masks with known content.
// Layout: row-major, extent(0) = nrow, extent(1) = ncol.
// roi convention: xmin/xmax track rows, ymin/ymax track columns.
// Empty mask sentinel: {INT_MAX, -1, INT_MAX, -1}.
// ============================================================================

class ComputeMaskRoiTest : public fdtest::GpuTest {
 protected:
  // Deliberately non-square to exercise the row-major flat-index unravel.
  static constexpr int NROW = 6;
  static constexpr int NCOL = 10;

  morpho::roi run(const std::vector<bool>& mask)
  {
    const auto sr = res().make_ctx();

    fdtest::device_buffer<bool> d_mask(res(), sr, mask);
    core::device_span2d<bool> view(d_mask.get(), NROW, NCOL);

    morpho::roi r = morpho::compute_mask_roi(sr, view);
    sr.wait();
    return r;
  }
};

TEST_F(ComputeMaskRoiTest, AllFalseReturnsEmptySentinel)
{
  std::vector<bool> mask(NROW * NCOL, false);
  auto r = run(mask);
  EXPECT_EQ(r.xmin, INT_MAX);
  EXPECT_EQ(r.xmax, -1);
  EXPECT_EQ(r.ymin, INT_MAX);
  EXPECT_EQ(r.ymax, -1);
}

TEST_F(ComputeMaskRoiTest, AllTrueCoversFullExtent)
{
  std::vector<bool> mask(NROW * NCOL, true);
  auto r = run(mask);
  EXPECT_EQ(r.xmin, 0);
  EXPECT_EQ(r.xmax, NROW - 1);
  EXPECT_EQ(r.ymin, 0);
  EXPECT_EQ(r.ymax, NCOL - 1);
}

TEST_F(ComputeMaskRoiTest, SinglePixelGivesDegenerateBox)
{
  // Single foreground pixel at (row=2, col=7) → xmin=xmax=2, ymin=ymax=7.
  std::vector<bool> mask(NROW * NCOL, false);
  const int row = 2, col = 7;
  mask[row * NCOL + col] = true;
  auto r = run(mask);
  EXPECT_EQ(r.xmin, row);
  EXPECT_EQ(r.xmax, row);
  EXPECT_EQ(r.ymin, col);
  EXPECT_EQ(r.ymax, col);
}

TEST_F(ComputeMaskRoiTest, RectangleBlockBoundingBox)
{
  // Block of true values in rows [1..4], cols [3..8].
  std::vector<bool> mask(NROW * NCOL, false);
  const int r0 = 1, r1 = 4, c0 = 3, c1 = 8;
  for (int r = r0; r <= r1; ++r)
    for (int c = c0; c <= c1; ++c) mask[r * NCOL + c] = true;
  auto r = run(mask);
  EXPECT_EQ(r.xmin, r0);
  EXPECT_EQ(r.xmax, r1);
  EXPECT_EQ(r.ymin, c0);
  EXPECT_EQ(r.ymax, c1);
}

TEST_F(ComputeMaskRoiTest, TwoDisjointPixelsSpanFullBox)
{
  // Two pixels far apart — bbox should cover both.
  std::vector<bool> mask(NROW * NCOL, false);
  mask[1 * NCOL + 2] = true;  // (row=1, col=2)
  mask[4 * NCOL + 9] = true;  // (row=4, col=9)
  auto r = run(mask);
  EXPECT_EQ(r.xmin, 1);
  EXPECT_EQ(r.xmax, 4);
  EXPECT_EQ(r.ymin, 2);
  EXPECT_EQ(r.ymax, 9);
}

TEST_F(ComputeMaskRoiTest, CornerPixelsHitBoundary)
{
  // Pixels at the four corners — bbox is the whole image.
  std::vector<bool> mask(NROW * NCOL, false);
  mask[0 * NCOL + 0] = true;
  mask[0 * NCOL + (NCOL - 1)] = true;
  mask[(NROW - 1) * NCOL + 0] = true;
  mask[(NROW - 1) * NCOL + (NCOL - 1)] = true;
  auto r = run(mask);
  EXPECT_EQ(r.xmin, 0);
  EXPECT_EQ(r.xmax, NROW - 1);
  EXPECT_EQ(r.ymin, 0);
  EXPECT_EQ(r.ymax, NCOL - 1);
}

// ============================================================================
// binary_dilation
//
// Tests focus on properties (extensivity, single-pixel oracle, translation
// equivariance) so they remain valid across implementation refactors.
// ============================================================================

class BinaryDilationTest : public fdtest::GpuTest {
 protected:
  // Non-square data shape to catch row/col indexing bugs.
  static constexpr int NROW = 6;
  static constexpr int NCOL = 10;

  // 3x3 square structuring element, anchor at center.
  static std::vector<bool> se_square_3x3() { return std::vector<bool>(9, true); }

  std::vector<uint8_t> run(const std::vector<bool>& data, const std::vector<bool>& se, int se_n, morpho::roi se_roi)
  {
    const auto sr = res().make_ctx();

    const std::size_t npix = NROW * NCOL;
    fdtest::device_buffer<bool> d_data(res(), sr, data);
    fdtest::device_buffer<bool> d_se(res(), sr, se);
    fdtest::device_buffer<bool> d_out(res(), sr, npix);
    CHECK_CUDA(cudaMemsetAsync(d_out.get(), 0, npix * sizeof(bool), sr.cuda_stream));

    core::device_span2d<bool> data_view(d_data.get(), NROW, NCOL);
    core::device_span2d<bool> se_view(d_se.get(), se_n, se_n);
    core::device_span2d<bool> out_view(d_out.get(), NROW, NCOL);

    morpho::binary_dilation(sr, data_view, se_view, se_roi, out_view);
    sr.wait();

    return d_out.to_host();
  }
};

TEST_F(BinaryDilationTest, AllFalseInputProducesAllFalse)
{
  std::vector<bool> data(NROW * NCOL, false);
  auto se = se_square_3x3();
  auto out = run(data, se, 3, morpho::roi{0, 3, 0, 3});
  for (uint8_t v : out) EXPECT_EQ(v, 0);
}

TEST_F(BinaryDilationTest, SinglePixelExpandsToStructuringElement)
{
  // Single foreground pixel at (row=3, col=5) dilated by a 3x3 square SE
  // should produce a 3x3 block of true centered at (3, 5); false elsewhere.
  std::vector<bool> data(NROW * NCOL, false);
  const int row = 3, col = 5;
  data[row * NCOL + col] = true;

  auto se = se_square_3x3();
  auto out = run(data, se, 3, morpho::roi{0, 3, 0, 3});

  for (int r = 0; r < NROW; ++r) {
    for (int c = 0; c < NCOL; ++c) {
      const bool inside_block = (r >= row - 1 && r <= row + 1 && c >= col - 1 && c <= col + 1);
      EXPECT_EQ(out[r * NCOL + c] != 0, inside_block) << "mismatch at (" << r << ", " << c << ")";
    }
  }
}

TEST_F(BinaryDilationTest, ExtensivityOutputContainsInput)
{
  // dilate(A) ⊇ A: every foreground pixel in input must remain foreground.
  std::vector<bool> data(NROW * NCOL, false);
  data[1 * NCOL + 2] = true;
  data[4 * NCOL + 8] = true;

  auto se = se_square_3x3();
  auto out = run(data, se, 3, morpho::roi{0, 3, 0, 3});

  for (std::size_t i = 0; i < data.size(); ++i) {
    if (data[i]) EXPECT_NE(out[i], 0) << "lost foreground at flat index " << i;
  }
}

TEST_F(BinaryDilationTest, TranslationEquivariance)
{
  // Shifting the input by (dr, dc) should shift the output by the same amount.
  // Catches indexing / off-by-one bugs (especially row vs col confusion in
  // non-square data).
  auto se = se_square_3x3();

  std::vector<bool> a(NROW * NCOL, false);
  a[2 * NCOL + 3] = true;
  auto out_a = run(a, se, 3, morpho::roi{0, 3, 0, 3});

  std::vector<bool> b(NROW * NCOL, false);
  const int dr = 1, dc = 4;
  b[(2 + dr) * NCOL + (3 + dc)] = true;
  auto out_b = run(b, se, 3, morpho::roi{0, 3, 0, 3});

  // Compare overlapping region: out_b[r, c] should equal out_a[r-dr, c-dc].
  for (int r = dr; r < NROW; ++r) {
    for (int c = dc; c < NCOL; ++c) {
      EXPECT_EQ(out_b[r * NCOL + c], out_a[(r - dr) * NCOL + (c - dc)])
          << "shift mismatch at (" << r << ", " << c << ")";
    }
  }
}
