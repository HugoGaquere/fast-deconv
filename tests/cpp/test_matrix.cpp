#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/matrix/stats.hpp>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/host_oracles.hpp"
#include "helpers/rng.hpp"

namespace core = fast_deconv::core;
namespace matrix = fast_deconv::matrix;
namespace fdtest = fast_deconv::test;

using fdtest::flat;

// Non-square, odd-sized image: the reductions span several blocks, the last one partial.
constexpr int kNrow = 33;
constexpr int kNcol = 47;
constexpr int kNpix = kNrow * kNcol;

// ============================================================================
// max_and_rms: max over the unmasked pixels, rms over all of them
// ============================================================================

class MaxAndRms : public fdtest::BackendTest {
 protected:
  // Zero image with a masked 9, an unmasked 3 and an unmasked -4.
  std::vector<float> img = std::vector<float>(kNpix, 0.0f);
  std::unique_ptr<bool[]> mask = std::make_unique<bool[]>(kNpix);
  const int masked = flat(10, 40, kNcol), positive = flat(20, 5, kNcol), negative = flat(2, 3, kNcol);

  void SetUp() override
  {
    img.at(masked) = 9.0f;
    mask[masked] = true;
    img.at(positive) = 3.0f;
    img.at(negative) = -4.0f;
  }

  auto stage_img(const core::exec_ctx& sr) const
  {
    return sr.stage(core::host_span2d<const float>(img.data(), kNrow, kNcol));
  }
  auto stage_mask(const core::exec_ctx& sr) const
  {
    return sr.stage(core::host_span2d<const bool>(mask.get(), kNrow, kNcol));
  }
};

// The masked 9 is not the max but does weigh in the rms.
TEST_F(MaxAndRms, MaskHidesPixelsFromTheMaxOnly)
{
  const auto sr = res().make_ctx();
  const auto d_mask = stage_mask(sr);

  const auto [max, rms] = matrix::max_and_rms(sr, stage_img(sr), d_mask, /*use_abs=*/false);
  EXPECT_FLOAT_EQ(max, 3.0f);
  EXPECT_NEAR(rms, fdtest::std_all(img), 1e-4f * fdtest::std_all(img));
}

TEST_F(MaxAndRms, UseAbsRanksByMagnitude)
{
  const auto sr = res().make_ctx();

  EXPECT_FLOAT_EQ(matrix::max_and_rms(sr, stage_img(sr), stage_mask(sr), /*use_abs=*/true).max, 4.0f);
}

// ============================================================================
// find_peak: argmax under a peak_criterion, carrying the signed pixel
// ============================================================================

class FindPeak : public fdtest::BackendTest {
 protected:
  matrix::peak run(const std::vector<float>& img, const bool* mask = nullptr, bool absolute = false)
  {
    const auto sr = res().make_ctx();
    const auto d_img = sr.stage(core::host_span2d<const float>(img.data(), kNrow, kNcol));
    std::optional<core::mdcontainer<const bool, 2>> d_mask;
    if (mask) d_mask = sr.stage(core::host_span2d<const bool>(mask, kNrow, kNcol));
    return matrix::find_peak(sr, d_img, {.mask = d_mask ? d_mask->data_handle() : nullptr, .absolute = absolute});
  }
};

// All-negative data catches a running max initialised to 0 instead of -inf, which would report 0.
TEST_F(FindPeak, FindsTheMaxOfAllNegativeData)
{
  std::mt19937 rng(67);
  std::vector<float> img(kNpix);
  fdtest::fill_uniform(rng, img, -2.0f, -1.0f);
  const int peak = flat(1, 2, kNcol);
  img.at(peak) = -0.5f;

  const auto p = run(img);
  EXPECT_EQ(p.index, peak);
  EXPECT_FLOAT_EQ(p.value, -0.5f);
}

// Symmetric sources give exact ties: the smaller flat index wins, so every backend picks the same pixel.
TEST_F(FindPeak, TieGoesToTheSmallerIndex)
{
  std::vector<float> img(kNpix, 0.0f);
  const int first = flat(3, 5, kNcol), second = flat(20, 1, kNcol);
  img.at(first) = 1.0f;
  img.at(second) = 1.0f;

  EXPECT_EQ(run(img).index, first);
}

TEST_F(FindPeak, MaskedPixelsNeverWin)
{
  std::vector<float> img(kNpix, 0.0f);
  const auto mask = std::make_unique<bool[]>(kNpix);
  const int masked = flat(4, 4, kNcol), unmasked = flat(25, 7, kNcol);
  img.at(masked) = 5.0f;
  mask[masked] = true;
  img.at(unmasked) = 2.0f;

  EXPECT_EQ(run(img, mask.get()).index, unmasked);
}

// The clean loop subtracts with signed_value, so abs ranks by magnitude but must keep the sign.
TEST_F(FindPeak, AbsRanksByMagnitudeAndKeepsTheSign)
{
  std::vector<float> img(kNpix, 0.0f);
  const int negative = flat(10, 30, kNcol);
  img.at(negative) = -3.0f;
  img.at(flat(25, 7, kNcol)) = 2.0f;

  const auto p = run(img, nullptr, /*absolute=*/true);
  EXPECT_EQ(p.index, negative);
  EXPECT_FLOAT_EQ(p.value, 3.0f);
  EXPECT_FLOAT_EQ(p.signed_value, -3.0f);
}

// A scale with no component is fully masked by the auto-mask: nothing qualifies, the peak is -inf.
TEST_F(FindPeak, FullyMaskedImageGivesNegativeInfinity)
{
  const auto mask = std::make_unique<bool[]>(kNpix);
  std::fill_n(mask.get(), kNpix, true);

  const auto p = run(std::vector<float>(kNpix, 1.0f), mask.get());
  EXPECT_TRUE(std::isinf(p.value) && p.value < 0.0f);
}
