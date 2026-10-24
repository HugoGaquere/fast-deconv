#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <fast_deconv/common/region.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/morphology/dilation.hpp>
#include <fast_deconv/morphology/roi.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "helpers/backend_test.hpp"

namespace common = fast_deconv::common;
namespace core = fast_deconv::core;
namespace morpho = fast_deconv::morphology;
namespace fdtest = fast_deconv::test;

// Non-square planes, so a row/col swap breaks every case.
constexpr int kNrow = 6;
constexpr int kNcol = 10;
constexpr int kNpix = kNrow * kNcol;

class Morphology : public fdtest::BackendTest {};

// Bounding box of the true pixels, half-open: rmax/cmax are one past the last one. Empty mask: all zero.
TEST_F(Morphology, MaskRoiIsTheBoundingBoxOfTruePixels)
{
  const struct {
    const char* name;
    std::vector<common::index2d> pixels;
    common::roi expected;
  } cases[] = {
      {"empty", {}, {0, 0, 0, 0}},
      {"single pixel", {{2, 7}}, {2, 3, 7, 8}},
      {"two disjoint pixels", {{1, 2}, {4, 9}}, {1, 5, 2, 10}},
      {"four corners", {{0, 0}, {0, kNcol - 1}, {kNrow - 1, 0}, {kNrow - 1, kNcol - 1}}, {0, kNrow, 0, kNcol}},
  };

  const auto sr = res().make_ctx();
  for (const auto& cs : cases) {
    const auto mask = std::make_unique<bool[]>(kNpix);
    for (const auto& [r, c] : cs.pixels) mask[r * kNcol + c] = true;
    const auto d_mask = sr.stage(core::host_span2d<const bool>(mask.get(), kNrow, kNcol));

    const auto got = morpho::compute_mask_roi(sr, d_mask);

    EXPECT_EQ(got.rmin, cs.expected.rmin) << cs.name;
    EXPECT_EQ(got.rmax, cs.expected.rmax) << cs.name;
    EXPECT_EQ(got.cmin, cs.expected.cmin) << cs.name;
    EXPECT_EQ(got.cmax, cs.expected.cmax) << cs.name;
  }
}

// A single pixel dilated by a 3x3 square becomes the 3x3 block around it, clipped at the border, and every
// other pixel is written false: build_auto_mask reuses the output buffer across scales, so it starts dirty.
TEST_F(Morphology, DilationOfSinglePixelIsTheClippedStructuringElement)
{
  const auto se = std::make_unique<bool[]>(3 * 3);
  std::fill_n(se.get(), 3 * 3, true);
  const auto sr = res().make_ctx();
  const auto d_se = sr.stage(core::host_span2d<const bool>(se.get(), 3, 3));

  for (const auto& [row, col] : {std::pair{3, 5}, {0, 0}, {0, kNcol - 1}, {kNrow - 1, 0}, {kNrow - 1, kNcol - 1}}) {
    const auto data = std::make_unique<bool[]>(kNpix);
    data[row * kNcol + col] = true;
    const auto out = std::make_unique<bool[]>(kNpix);
    std::fill_n(out.get(), kNpix, true);
    const auto d_data = sr.stage(core::host_span2d<const bool>(data.get(), kNrow, kNcol));
    auto d_out = sr.stage(core::host_span2d<bool>(out.get(), kNrow, kNcol));

    morpho::binary_dilation(sr, d_data, d_se, common::roi{0, 3, 0, 3}, d_out);
    sr.unstage(d_out, out.get());
    sr.wait();

    for (int r = 0; r < kNrow; ++r)
      for (int c = 0; c < kNcol; ++c) {
        const bool in_block = std::abs(r - row) <= 1 && std::abs(c - col) <= 1;
        EXPECT_EQ(out[r * kNcol + c], in_block)
            << "pixel (" << row << ", " << col << "): mismatch at (" << r << ", " << c << ")";
      }
  }
}
