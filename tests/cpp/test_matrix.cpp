#include <gtest/gtest.h>

#include <cmath>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/matrix/stats.hpp>
#include <optional>
#include <random>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/device_buffers.hpp"
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
// stats_ctx::max_and_rms: max over the unmasked pixels, rms over all of them
// ============================================================================

class MaxAndRms : public fdtest::BackendTest {
 protected:
  // Zero image with a masked 9, an unmasked 3 and an unmasked -4.
  std::vector<float> img = std::vector<float>(kNpix, 0.0f);
  std::vector<bool> mask = std::vector<bool>(kNpix, false);
  const int masked = flat(10, 40, kNcol), positive = flat(20, 5, kNcol), negative = flat(2, 3, kNcol);

  void SetUp() override
  {
    img.at(masked) = 9.0f;
    mask.at(masked) = true;
    img.at(positive) = 3.0f;
    img.at(negative) = -4.0f;
  }
};

// The masked 9 is not the max but does weigh in the rms. The driver reuses one ctx for every cycle, so a
// second call on new data must see that data.
TEST_F(MaxAndRms, MaskHidesPixelsFromTheMaxOnly)
{
  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_img(sr, img);
  fdtest::device_buffer<bool> d_mask(sr, mask);
  const core::span2d<const float> img_view(d_img.get(), kNrow, kNcol);
  const core::span2d<const bool> mask_view(d_mask.get(), kNrow, kNcol);
  matrix::stats_ctx stats(sr, kNpix, /*use_abs=*/false);

  const auto [max, rms] = stats.max_and_rms(img_view, mask_view);
  EXPECT_FLOAT_EQ(max, 3.0f);
  EXPECT_NEAR(rms, fdtest::std_all(img), 1e-4f * fdtest::std_all(img));

  img.at(positive) = 5.0f;
  d_img.from_host(img);
  EXPECT_FLOAT_EQ(stats.max_and_rms(img_view, mask_view).max, 5.0f);
}

TEST_F(MaxAndRms, UseAbsRanksByMagnitude)
{
  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_img(sr, img);
  fdtest::device_buffer<bool> d_mask(sr, mask);
  matrix::stats_ctx stats(sr, kNpix, /*use_abs=*/true);

  const auto s = stats.max_and_rms(core::span2d<const float>(d_img.get(), kNrow, kNcol),
                                   core::span2d<const bool>(d_mask.get(), kNrow, kNcol));
  EXPECT_FLOAT_EQ(s.max, 4.0f);
}

// ============================================================================
// find_peak: argmax under a peak_criterion, carrying the signed pixel
// ============================================================================

class FindPeak : public fdtest::BackendTest {
 protected:
  // An empty @p mask means no mask at all (a null criterion mask).
  matrix::peak run(const std::vector<float>& img, const std::vector<bool>& mask = {}, bool absolute = false)
  {
    const auto sr = res().make_ctx();
    fdtest::device_buffer<float> d_img(sr, img);
    std::optional<fdtest::device_buffer<bool>> d_mask;
    if (!mask.empty()) d_mask.emplace(sr, mask);
    const matrix::peak_criterion criterion{.mask = d_mask ? d_mask->get() : nullptr, .absolute = absolute};
    return matrix::find_peak(sr, core::span2d<const float>(d_img.get(), kNrow, kNcol), criterion);
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
  std::vector<bool> mask(kNpix, false);
  const int masked = flat(4, 4, kNcol), unmasked = flat(25, 7, kNcol);
  img.at(masked) = 5.0f;
  mask.at(masked) = true;
  img.at(unmasked) = 2.0f;

  EXPECT_EQ(run(img, mask).index, unmasked);
}

// The clean loop subtracts with signed_value, so abs ranks by magnitude but must keep the sign.
TEST_F(FindPeak, AbsRanksByMagnitudeAndKeepsTheSign)
{
  std::vector<float> img(kNpix, 0.0f);
  const int negative = flat(10, 30, kNcol);
  img.at(negative) = -3.0f;
  img.at(flat(25, 7, kNcol)) = 2.0f;

  const auto p = run(img, {}, /*absolute=*/true);
  EXPECT_EQ(p.index, negative);
  EXPECT_FLOAT_EQ(p.value, 3.0f);
  EXPECT_FLOAT_EQ(p.signed_value, -3.0f);
}

// A scale with no component is fully masked by the auto-mask: nothing qualifies, the peak is -inf.
TEST_F(FindPeak, FullyMaskedImageGivesNegativeInfinity)
{
  const auto p = run(std::vector<float>(kNpix, 1.0f), std::vector<bool>(kNpix, true));
  EXPECT_TRUE(std::isinf(p.value) && p.value < 0.0f);
}
