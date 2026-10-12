#include <gtest/gtest.h>

#include <cfloat>
#include <cmath>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/matrix/stats.hpp>
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

namespace {

// Non-square, non-power-of-two image with a ~20% random mask (mask true = excluded).
constexpr int kNrow = 33;
constexpr int kNcol = 47;
constexpr int kNpix = kNrow * kNcol;

std::vector<float> make_image(std::mt19937& rng)
{
  std::vector<float> img(kNpix);
  fdtest::fill_uniform(rng, img, -1.0f, 1.0f);
  return img;
}

std::vector<uint8_t> make_mask(std::mt19937& rng)
{
  std::vector<uint8_t> mask(kNpix);
  for (auto& m : mask) m = fdtest::uniform01(rng) < 0.2f ? 1 : 0;
  return mask;
}

std::vector<bool> to_bool(const std::vector<uint8_t>& mask)
{
  std::vector<bool> b(mask.size());
  for (std::size_t i = 0; i < mask.size(); ++i) b.at(i) = mask.at(i) != 0;
  return b;
}

}  // namespace

class MatrixReductions : public fdtest::BackendTest {};

// stats_ctx::run is deliberately asymmetric (stats.hpp): the mask excludes
// pixels from the MAX only; the rms is over every pixel. Pin that contract.
TEST_F(MatrixReductions, ComputeStatsMasksMaxButNotRms)
{
  std::mt19937 rng(47);
  auto img = make_image(rng);
  auto mask = make_mask(rng);
  const int planted = flat(10, 40, kNcol);
  img.at(planted) = 9.0f;
  mask.at(planted) = 1;

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_img(sr, img);
  fdtest::device_buffer<bool> d_mask(sr, to_bool(mask));
  core::span2d<float> img_view(d_img.get(), kNrow, kNcol);
  core::span2d<bool> mask_view(d_mask.get(), kNrow, kNcol);

  matrix::stats_ctx ws(sr, kNpix, /*use_abs=*/false);
  const auto [max_v, rms_v] = ws.run(img_view, mask_view);

  EXPECT_FLOAT_EQ(max_v, fdtest::masked_max(img, mask, false));
  const float rms_expected = fdtest::std_all(img);  // mask ignored, planted 9.0 included
  EXPECT_NEAR(rms_v, rms_expected, 1e-4f * rms_expected);
}

TEST_F(MatrixReductions, ComputeStatsWithAbsAndWorkspaceReuse)
{
  std::mt19937 rng(53);
  auto img1 = make_image(rng);
  img1.at(flat(2, 3, kNcol)) = -4.0f;  // abs-extreme is negative
  auto img2 = make_image(rng);
  img2.at(flat(30, 44, kNcol)) = -2.5f;

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_img(sr, img1);
  fdtest::device_buffer<bool> d_mask(sr, std::vector<bool>(kNpix, false));
  core::span2d<float> img_view(d_img.get(), kNrow, kNcol);
  core::span2d<bool> mask_view(d_mask.get(), kNrow, kNcol);

  // use_abs is fixed at construction; the same ctx must serve any number
  // of calls (temp bytes are queried once in the ctor).
  matrix::stats_ctx ws(sr, kNpix, /*use_abs=*/true);

  const auto s1 = ws.run(img_view, mask_view);
  EXPECT_FLOAT_EQ(s1.max, 4.0f);
  EXPECT_NEAR(s1.rms, fdtest::std_all(img1), 1e-4f * fdtest::std_all(img1));

  d_img.from_host(img2);
  const auto s2 = ws.run(img_view, mask_view);
  EXPECT_FLOAT_EQ(s2.max, 2.5f);
  EXPECT_NEAR(s2.rms, fdtest::std_all(img2), 1e-4f * fdtest::std_all(img2));
}

// The async variant leaves the packed accumulator on device; finalize on the
// host from stats_acc and compare against the oracles.
TEST_F(MatrixReductions, ComputeStatsAsyncLeavesResultOnDevice)
{
  std::mt19937 rng(59);
  const auto img = make_image(rng);

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_img(sr, img);
  fdtest::device_buffer<bool> d_mask(sr, std::vector<bool>(kNpix, false));
  core::span2d<float> img_view(d_img.get(), kNrow, kNcol);
  core::span2d<bool> mask_view(d_mask.get(), kNrow, kNcol);

  matrix::stats_ctx ws(sr, kNpix, /*use_abs=*/false);
  ws.run_async(img_view, mask_view);

  matrix::stats_acc acc{};
  sr.copy(&acc, ws.device_state(), 1);
  sr.wait();

  ASSERT_EQ(acc.count, kNpix);
  EXPECT_FLOAT_EQ(acc.max_v, fdtest::masked_max(img, std::vector<uint8_t>(kNpix, 0), false));
  const double mean = static_cast<double>(acc.sum) / acc.count;
  const float rms = static_cast<float>(std::sqrt(std::max(acc.sum_sq / acc.count - mean * mean, 0.0)));
  const float expected = fdtest::std_all(img);
  EXPECT_NEAR(rms, expected, 1e-4f * expected);
}

// ============================================================================
// matrix::find_peak — argmax of a peak_criterion, carrying the signed pixel
// ============================================================================

class FindPeak : public fdtest::BackendTest {
 protected:
  matrix::peak run(const core::exec_ctx& sr, const std::vector<float>& img, const std::vector<uint8_t>* mask,
                   bool absolute)
  {
    fdtest::device_buffer<float> d_img(sr, img);
    fdtest::device_buffer<bool> d_mask(sr, mask ? to_bool(*mask) : std::vector<bool>(1));
    const matrix::peak_criterion criterion{.mask = mask ? d_mask.get() : nullptr, .absolute = absolute};
    return matrix::find_peak(sr, core::span2d<const float>(d_img.get(), kNrow, kNcol), criterion);
  }
};

// An empty criterion is a plain argmax. All-negative data catches a running max
// initialised to 0 instead of -inf, which would report 0 at a bogus index.
TEST_F(FindPeak, EmptyCriterionIsPlainArgmaxOnAllNegativeData)
{
  std::mt19937 rng(67);
  std::vector<float> img(kNpix);
  fdtest::fill_uniform(rng, img, -2.0f, -1.0f);
  const int a = flat(1, 2, kNcol);
  img.at(a) = -0.5f;  // unique max, still negative

  const auto sr = res().make_ctx();
  const auto p = run(sr, img, nullptr, /*absolute=*/false);
  EXPECT_EQ(p.index, a);
  EXPECT_FLOAT_EQ(p.value, -0.5f);
  EXPECT_FLOAT_EQ(p.signed_value, -0.5f);
}

// Equal maxima: the smaller flat index wins, on every backend and thread count.
TEST_F(FindPeak, TieGoesToTheSmallerIndex)
{
  std::vector<float> img(kNpix, 0.0f);
  const int first = flat(3, 5, kNcol), second = flat(20, 1, kNcol);
  img.at(second) = 1.0f;
  img.at(first) = 1.0f;

  const auto sr = res().make_ctx();
  EXPECT_EQ(run(sr, img, nullptr, /*absolute=*/false).index, first);
}

// Masked pixels never win, abs ranks a negative pixel by its magnitude, and the
// winner's sign survives in signed_value: the clean loop subtracts with it.
TEST_F(FindPeak, CriterionMasksRanksByAbsAndKeepsTheSign)
{
  std::vector<float> img(kNpix, 0.0f);
  std::vector<uint8_t> mask(kNpix, 0);
  const int masked = flat(4, 4, kNcol), negative = flat(10, 30, kNcol), positive = flat(25, 7, kNcol);
  img.at(masked) = 5.0f;
  mask.at(masked) = 1;
  img.at(negative) = -3.0f;
  img.at(positive) = 2.0f;

  const auto sr = res().make_ctx();

  const auto by_abs = run(sr, img, &mask, /*absolute=*/true);
  EXPECT_EQ(by_abs.index, negative);
  EXPECT_FLOAT_EQ(by_abs.value, 3.0f);
  EXPECT_FLOAT_EQ(by_abs.signed_value, -3.0f);

  const auto signed_rank = run(sr, img, &mask, /*absolute=*/false);
  EXPECT_EQ(signed_rank.index, positive);
  EXPECT_FLOAT_EQ(signed_rank.value, 2.0f);
  EXPECT_FLOAT_EQ(signed_rank.signed_value, 2.0f);
}

// Random image and mask against the host oracle, both abs settings.
TEST_F(FindPeak, MatchesMaskedMaxOracle)
{
  std::mt19937 rng(71);
  const auto img = make_image(rng);
  const auto mask = make_mask(rng);
  const auto sr = res().make_ctx();

  for (const bool absolute : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "absolute=" << absolute);
    const auto p = run(sr, img, &mask, absolute);
    ASSERT_GE(p.index, 0);
    ASSERT_LT(p.index, kNpix);
    EXPECT_EQ(mask.at(p.index), 0);
    EXPECT_FLOAT_EQ(p.value, fdtest::masked_max(img, mask, absolute));
    EXPECT_FLOAT_EQ(p.signed_value, img.at(p.index));
  }
}

// Every pixel masked: nothing qualifies, so the peak value is -inf.
TEST_F(FindPeak, FullyMaskedPlaneReportsNegativeInfinity)
{
  std::mt19937 rng(73);
  const auto img = make_image(rng);
  const std::vector<uint8_t> mask(kNpix, 1);

  const auto sr = res().make_ctx();
  const auto p = run(sr, img, &mask, /*absolute=*/true);
  EXPECT_TRUE(std::isinf(p.value) && p.value < 0.0f);
}
