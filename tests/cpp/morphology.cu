#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <climits>
#include <vector>

#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/morphology/dilation.hpp>
#include <fast_deconv/morphology/roi.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace core = fast_deconv::core;
namespace morpho = fast_deconv::morphology;

// ============================================================================
// Helpers
// ============================================================================

namespace {

// Upload a host bool buffer to device, returns the device pointer (caller frees).
bool* upload_bool(const core::resources& res, const core::stream_resources& sr,
                  const std::vector<bool>& host, std::size_t n)
{
  // std::vector<bool> is bit-packed, so unpack into a uint8_t buffer first.
  std::vector<uint8_t> packed(n);
  for (std::size_t i = 0; i < n; ++i) packed[i] = host[i] ? 1 : 0;

  bool* d_ptr = res.alloc_async<bool>(n, sr);
  CHECK_CUDA(cudaMemcpyAsync(d_ptr, packed.data(), n * sizeof(bool), cudaMemcpyHostToDevice,
                             sr.cuda_stream));
  sr.sync();
  return d_ptr;
}

std::vector<uint8_t> download_bool(const core::stream_resources& sr, const bool* d_ptr,
                                   std::size_t n)
{
  // Read into uint8_t (1 byte per element) to avoid std::vector<bool> bit-packing.
  std::vector<uint8_t> host(n);
  CHECK_CUDA(cudaMemcpyAsync(host.data(), d_ptr, n * sizeof(bool), cudaMemcpyDeviceToHost,
                             sr.cuda_stream));
  sr.sync();
  return host;
}

}  // namespace

// ============================================================================
// compute_mask_roi
//
// Property-based oracle tests on small non-square masks with known content.
// Layout: row-major, extent(0) = nrow, extent(1) = ncol.
// roi convention: xmin/xmax track rows, ymin/ymax track columns.
// Empty mask sentinel: {INT_MAX, -1, INT_MAX, -1}.
// ============================================================================

class ComputeMaskRoiTest : public ::testing::Test {
 protected:
  // Deliberately non-square to exercise the row-major flat-index unravel.
  static constexpr int NROW = 6;
  static constexpr int NCOL = 10;

  morpho::roi run(const std::vector<bool>& mask)
  {
    core::resources resources(0);
    const auto& sr = resources.get_stream_resources();

    bool* d_mask = upload_bool(resources, sr, mask, NROW * NCOL);
    core::device_span2d<bool> view(d_mask, NROW, NCOL);

    morpho::roi r = morpho::compute_mask_roi(sr, view);

    resources.free_async(d_mask, sr);
    sr.sync();
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
// NOTE: binary_dilation is currently a stub. Tests are DISABLED_ until the
// kernel is implemented. Re-enable by removing the DISABLED_ prefix.
//
// Tests focus on properties (extensivity, single-pixel oracle, translation
// equivariance) so they remain valid across implementation refactors.
// ============================================================================

class BinaryDilationTest : public ::testing::Test {
 protected:
  // Non-square data shape to catch row/col indexing bugs.
  static constexpr int NROW = 6;
  static constexpr int NCOL = 10;

  // 3x3 square structuring element, anchor at center.
  static std::vector<bool> se_square_3x3() { return std::vector<bool>(9, true); }

  std::vector<uint8_t> run(const std::vector<bool>& data, const std::vector<bool>& se,
                           int se_n, morpho::roi se_roi)
  {
    core::resources resources(0);
    const auto& sr = resources.get_stream_resources();

    const std::size_t npix = NROW * NCOL;
    bool* d_data = upload_bool(resources, sr, data, npix);
    bool* d_se = upload_bool(resources, sr, se, se_n * se_n);
    bool* d_out = resources.alloc_async<bool>(npix, sr);
    CHECK_CUDA(cudaMemsetAsync(d_out, 0, npix * sizeof(bool), sr.cuda_stream));
    sr.sync();

    core::device_span2d<bool> data_view(d_data, NROW, NCOL);
    core::device_span2d<bool> se_view(d_se, se_n, se_n);
    core::device_span2d<bool> out_view(d_out, NROW, NCOL);

    morpho::binary_dilation(sr, data_view, se_view, se_roi, out_view);
    sr.sync();

    auto host_out = download_bool(sr, d_out, npix);

    resources.free_async(d_data, sr);
    resources.free_async(d_se, sr);
    resources.free_async(d_out, sr);
    sr.sync();
    return host_out;
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
      const bool inside_block =
          (r >= row - 1 && r <= row + 1 && c >= col - 1 && c <= col + 1);
      EXPECT_EQ(out[r * NCOL + c] != 0, inside_block)
          << "mismatch at (" << r << ", " << c << ")";
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
