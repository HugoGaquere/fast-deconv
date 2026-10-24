#include <gtest/gtest.h>

#include <cstdlib>
#include <fast_deconv/common/region.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/morphology/dilation.hpp>
#include <fast_deconv/morphology/roi.hpp>
#include <utility>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/device_buffers.hpp"

namespace common = fast_deconv::common;
namespace core = fast_deconv::core;
namespace morpho = fast_deconv::morphology;
namespace fdtest = fast_deconv::test;

// Non-square planes, so a row/col swap breaks every case.
constexpr int kNrow = 6;
constexpr int kNcol = 10;

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
    std::vector<bool> mask(kNrow * kNcol, false);
    for (const auto& [r, c] : cs.pixels) mask.at(r * kNcol + c) = true;
    fdtest::device_buffer<bool> d_mask(sr, mask);

    const auto got = morpho::compute_mask_roi(sr, core::span2d<const bool>(d_mask.get(), kNrow, kNcol));

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
  const auto sr = res().make_ctx();
  fdtest::device_buffer<bool> d_se(sr, std::vector<bool>(3 * 3, true));

  for (const auto& [row, col] : {std::pair{3, 5}, {0, 0}, {0, kNcol - 1}, {kNrow - 1, 0}, {kNrow - 1, kNcol - 1}}) {
    std::vector<bool> data(kNrow * kNcol, false);
    data.at(row * kNcol + col) = true;
    fdtest::device_buffer<bool> d_data(sr, data), d_out(sr, data.size());
    d_out.fill_bytes(0xff);

    morpho::binary_dilation(sr, core::span2d<const bool>(d_data.get(), kNrow, kNcol),
                            core::span2d<const bool>(d_se.get(), 3, 3), common::roi{0, 3, 0, 3},
                            core::span2d<bool>(d_out.get(), kNrow, kNcol));

    const auto out = d_out.to_host();
    for (int r = 0; r < kNrow; ++r)
      for (int c = 0; c < kNcol; ++c) {
        const bool in_block = std::abs(r - row) <= 1 && std::abs(c - col) <= 1;
        EXPECT_EQ(out.at(r * kNcol + c), in_block ? 1 : 0)
            << "pixel (" << row << ", " << col << "): mismatch at (" << r << ", " << c << ")";
      }
  }
}
