#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cfloat>
#include <cmath>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/matrix/max.hpp>
#include <fast_deconv/matrix/rms.hpp>
#include <fast_deconv/matrix/stats.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <random>
#include <vector>

#include "helpers/device_buffers.hpp"
#include "helpers/gpu_test.hpp"
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

class MatrixReductions : public fdtest::GpuTest {};

// The mask excludes the true global max, so the reduction must return the best
// UNMASKED value — a max that ignores the mask fails here.
TEST_F(MatrixReductions, MaxHonorsMask)
{
  std::mt19937 rng(31);
  auto img = make_image(rng);
  auto mask = make_mask(rng);

  const int planted = flat(20, 30, kNcol);
  img.at(planted) = 9.0f;
  mask.at(planted) = 1;  // exclude the global max

  const auto sr = res().make_stream();
  fdtest::device_buffer<float> d_img(res(), sr, img);
  fdtest::device_buffer<bool> d_mask(res(), sr, to_bool(mask));
  core::device_span2d<float> img_view(d_img.get(), kNrow, kNcol);
  core::device_span2d<bool> mask_view(d_mask.get(), kNrow, kNcol);

  const float got = matrix::max(sr, img_view, mask_view, /*use_abs=*/false);
  EXPECT_FLOAT_EQ(got, fdtest::masked_max(img, mask, false));
  EXPECT_LT(got, 9.0f);
}

TEST_F(MatrixReductions, MaxWithAbsPicksNegativeExtreme)
{
  std::mt19937 rng(37);
  auto img = make_image(rng);
  const std::vector<uint8_t> mask(kNpix, 0);

  img.at(flat(5, 5, kNcol)) = -3.0f;  // extreme value is negative

  const auto sr = res().make_stream();
  fdtest::device_buffer<float> d_img(res(), sr, img);
  fdtest::device_buffer<bool> d_mask(res(), sr, std::vector<bool>(kNpix, false));
  core::device_span2d<float> img_view(d_img.get(), kNrow, kNcol);
  core::device_span2d<bool> mask_view(d_mask.get(), kNrow, kNcol);

  EXPECT_FLOAT_EQ(matrix::max(sr, img_view, mask_view, /*use_abs=*/true), 3.0f);
  EXPECT_FLOAT_EQ(matrix::max(sr, img_view, mask_view, /*use_abs=*/false), fdtest::masked_max(img, mask, false));
}

TEST_F(MatrixReductions, RmsMatchesMaskedStdOracle)
{
  std::mt19937 rng(41);
  const auto img = make_image(rng);
  const auto mask = make_mask(rng);

  const auto sr = res().make_stream();
  fdtest::device_buffer<float> d_img(res(), sr, img);
  fdtest::device_buffer<bool> d_mask(res(), sr, to_bool(mask));
  core::device_span2d<float> img_view(d_img.get(), kNrow, kNcol);
  core::device_span2d<bool> mask_view(d_mask.get(), kNrow, kNcol);

  const float got = matrix::rms(sr, img_view, mask_view);
  const float expected = fdtest::std_unmasked(img, mask);
  EXPECT_NEAR(got, expected, 1e-5f * std::fabs(expected) + 1e-7f);
}

TEST_F(MatrixReductions, RmsAllMaskedReturnsZero)
{
  std::mt19937 rng(43);
  const auto img = make_image(rng);

  const auto sr = res().make_stream();
  fdtest::device_buffer<float> d_img(res(), sr, img);
  fdtest::device_buffer<bool> d_mask(res(), sr, std::vector<bool>(kNpix, true));
  core::device_span2d<float> img_view(d_img.get(), kNrow, kNcol);
  core::device_span2d<bool> mask_view(d_mask.get(), kNrow, kNcol);

  EXPECT_FLOAT_EQ(matrix::rms(sr, img_view, mask_view), 0.0f);
}

// compute_stats is deliberately asymmetric (stats.hpp): the mask excludes
// pixels from the MAX only; the rms is over every pixel. Pin that contract.
TEST_F(MatrixReductions, ComputeStatsMasksMaxButNotRms)
{
  std::mt19937 rng(47);
  auto img = make_image(rng);
  auto mask = make_mask(rng);
  const int planted = flat(10, 40, kNcol);
  img.at(planted) = 9.0f;
  mask.at(planted) = 1;

  const auto sr = res().make_stream();
  fdtest::device_buffer<float> d_img(res(), sr, img);
  fdtest::device_buffer<bool> d_mask(res(), sr, to_bool(mask));
  core::device_span2d<float> img_view(d_img.get(), kNrow, kNcol);
  core::device_span2d<bool> mask_view(d_mask.get(), kNrow, kNcol);

  matrix::stats_workspace ws(sr, kNpix, /*use_abs=*/false);
  const auto [max_v, rms_v] = matrix::compute_stats(ws, img_view, mask_view);

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

  const auto sr = res().make_stream();
  fdtest::device_buffer<float> d_img(res(), sr, img1);
  fdtest::device_buffer<bool> d_mask(res(), sr, std::vector<bool>(kNpix, false));
  core::device_span2d<float> img_view(d_img.get(), kNrow, kNcol);
  core::device_span2d<bool> mask_view(d_mask.get(), kNrow, kNcol);

  // use_abs is fixed at construction; the same workspace must serve any number
  // of calls (temp bytes are queried once in the ctor).
  matrix::stats_workspace ws(sr, kNpix, /*use_abs=*/true);

  const auto s1 = matrix::compute_stats(ws, img_view, mask_view);
  EXPECT_FLOAT_EQ(s1.max, 4.0f);
  EXPECT_NEAR(s1.rms, fdtest::std_all(img1), 1e-4f * fdtest::std_all(img1));

  d_img.from_host(img2);
  const auto s2 = matrix::compute_stats(ws, img_view, mask_view);
  EXPECT_FLOAT_EQ(s2.max, 2.5f);
  EXPECT_NEAR(s2.rms, fdtest::std_all(img2), 1e-4f * fdtest::std_all(img2));
}

// The async variant leaves the packed accumulator on device; finalize on the
// host from stats_acc and compare against the oracles.
TEST_F(MatrixReductions, ComputeStatsAsyncLeavesResultOnDevice)
{
  std::mt19937 rng(59);
  const auto img = make_image(rng);

  const auto sr = res().make_stream();
  fdtest::device_buffer<float> d_img(res(), sr, img);
  fdtest::device_buffer<bool> d_mask(res(), sr, std::vector<bool>(kNpix, false));
  core::device_span2d<float> img_view(d_img.get(), kNrow, kNcol);
  core::device_span2d<bool> mask_view(d_mask.get(), kNrow, kNcol);

  matrix::stats_workspace ws(sr, kNpix, /*use_abs=*/false);
  matrix::compute_stats_async(ws, img_view, mask_view);

  matrix::stats_acc acc{};
  CHECK_CUDA(cudaMemcpyAsync(&acc, ws.d_state, sizeof(acc), cudaMemcpyDeviceToHost, sr.cuda_stream));
  sr.sync();

  ASSERT_EQ(acc.count, kNpix);
  EXPECT_FLOAT_EQ(acc.max_v, fdtest::masked_max(img, std::vector<uint8_t>(kNpix, 0), false));
  const double mean = static_cast<double>(acc.sum) / acc.count;
  const float rms = static_cast<float>(std::sqrt(std::max(acc.sum_sq / acc.count - mean * mean, 0.0)));
  const float expected = fdtest::std_all(img);
  EXPECT_NEAR(rms, expected, 1e-4f * expected);
}

// ============================================================================
// matrix::argmax (CUB DeviceReduce::ArgMax wrapper)
// ============================================================================

class ArgmaxWorkspace : public fdtest::GpuTest {};

TEST_F(ArgmaxWorkspace, FindsPlantedUniquePeak)
{
  std::mt19937 rng(61);
  std::vector<float> img(kNpix);
  fdtest::fill_uniform(rng, img, 0.0f, 1.0f);
  const int planted = flat(17, 23, kNcol);
  img.at(planted) = 5.0f;

  const auto sr = res().make_stream();
  fdtest::device_buffer<float> d_img(res(), sr, img);

  matrix::argmax_workspace ws(sr, kNpix);
  const auto [val, idx] = matrix::argmax(ws, d_img.get());
  EXPECT_FLOAT_EQ(val, 5.0f);
  EXPECT_EQ(idx, planted);
}

TEST_F(ArgmaxWorkspace, ReusableAcrossCallsAndAllNegativeSafe)
{
  std::mt19937 rng(67);
  std::vector<float> img(kNpix);
  fdtest::fill_uniform(rng, img, -2.0f, -1.0f);
  const int a = flat(1, 2, kNcol);
  img.at(a) = -0.5f;  // unique max, still negative

  const auto sr = res().make_stream();
  fdtest::device_buffer<float> d_img(res(), sr, img);
  matrix::argmax_workspace ws(sr, kNpix);

  const auto [v1, i1] = matrix::argmax(ws, d_img.get());
  EXPECT_FLOAT_EQ(v1, -0.5f);
  EXPECT_EQ(i1, a);

  // Move the peak and reuse the same workspace.
  img.at(a) = -1.5f;
  const int b = flat(28, 3, kNcol);
  img.at(b) = -0.25f;
  d_img.from_host(img);

  const auto [v2, i2] = matrix::argmax(ws, d_img.get());
  EXPECT_FLOAT_EQ(v2, -0.25f);
  EXPECT_EQ(i2, b);
}
