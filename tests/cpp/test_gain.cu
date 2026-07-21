#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <fast_deconv/common/gain.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <random>
#include <vector>

#include "helpers/device_buffers.hpp"
#include "helpers/gpu_test.hpp"
#include "helpers/host_oracles.hpp"
#include "helpers/rng.hpp"

namespace core = fast_deconv::core;
namespace common = fast_deconv::common;
namespace fdtest = fast_deconv::test;

namespace {

constexpr int kFacets = 2;
constexpr int kFreq = 3;
constexpr int kPsfH = 9;
constexpr int kPsfW = 9;
constexpr int kPsfNpix = kPsfH * kPsfW;
constexpr float kGamma = 0.1f;

// gain[facet] = gamma / max_pixel( sum_f w[f] * psf[facet, f] )
std::vector<float> host_gains(const std::vector<float>& psfs, const std::vector<float>& weights, int n_facets)
{
  std::vector<float> gains(n_facets);
  for (int b = 0; b < n_facets; ++b) {
    const std::vector<float> facet(psfs.begin() + b * kFreq * kPsfNpix, psfs.begin() + (b + 1) * kFreq * kPsfNpix);
    const auto wmean = fdtest::weighted_sum(facet, weights, kPsfNpix);
    gains.at(b) = kGamma / fdtest::masked_max(wmean, std::vector<uint8_t>(kPsfNpix, 0), false);
  }
  return gains;
}

}  // namespace

class GainBatched : public fdtest::GpuTest {};

TEST_F(GainBatched, PerFacetGainMatchesWeightedMeanMaxOracle)
{
  std::mt19937 rng(89);
  std::vector<float> psfs(kFacets * kFreq * kPsfNpix);
  fdtest::fill_uniform(rng, psfs, 0.0f, 0.5f);
  // Distinct known peaks so the two facets produce clearly different gains.
  psfs.at(0 * kFreq * kPsfNpix + 0 * kPsfNpix + fdtest::flat(4, 4, kPsfW)) = 1.0f;
  psfs.at(1 * kFreq * kPsfNpix + 1 * kPsfNpix + fdtest::flat(2, 6, kPsfW)) = 2.0f;
  const std::vector<float> weights = {0.5f, 0.3f, 0.2f};

  const auto sr = res().make_stream();
  fdtest::device_buffer<float> d_psfs(res(), sr, psfs);
  fdtest::device_buffer<float> d_w(res(), sr, weights);

  core::device_span4d<float> psf_view(d_psfs.get(), kFacets, kFreq, kPsfH, kPsfW);
  core::device_vect<float> w_view(d_w.get(), kFreq);

  const auto gains = common::compute_gain_batched(sr, psf_view, w_view, kGamma);
  const auto expected = host_gains(psfs, weights, kFacets);

  ASSERT_EQ(gains.size(), static_cast<std::size_t>(kFacets));
  for (int b = 0; b < kFacets; ++b) EXPECT_NEAR(gains.at(b), expected.at(b), 1e-6f * expected.at(b)) << "facet " << b;
}

// compute_all_gains_batched: scale 0 short-circuits to gamma; other scales go
// through the per-facet oracle. The output ordering gains[s * n_facets + f] is
// what run_ddmsc_cycles indexes into — pin it.
TEST_F(GainBatched, AllGainsScaleZeroFastPathAndScaleMajorOrdering)
{
  const int n_scales = 3;
  std::mt19937 rng(97);
  std::vector<float> psfs(n_scales * kFacets * kFreq * kPsfNpix);
  fdtest::fill_uniform(rng, psfs, 0.1f, 0.5f);
  // Per-(scale, facet) distinct peaks so any ordering mixup breaks a value.
  for (int s = 0; s < n_scales; ++s)
    for (int b = 0; b < kFacets; ++b)
      psfs.at(((s * kFacets + b) * kFreq + 0) * kPsfNpix + fdtest::flat(4, 4, kPsfW)) = 1.0f + s + 0.5f * b;
  const std::vector<float> weights = {0.5f, 0.3f, 0.2f};

  const auto sr = res().make_stream();
  fdtest::device_buffer<float> d_psfs(res(), sr, psfs);
  fdtest::device_buffer<float> d_w(res(), sr, weights);

  core::device_span5d<float> psf_view(d_psfs.get(), n_scales, kFacets, kFreq, kPsfH, kPsfW);
  core::device_vect<float> w_view(d_w.get(), kFreq);

  const auto all_gains = common::compute_all_gains_batched(sr, psf_view, w_view, kGamma);
  ASSERT_EQ(all_gains.size(), static_cast<std::size_t>(n_scales * kFacets));

  // Scale 0: exactly gamma, no reduction involved.
  EXPECT_FLOAT_EQ(all_gains.at(0), kGamma);
  EXPECT_FLOAT_EQ(all_gains.at(1), kGamma);

  for (int s = 1; s < n_scales; ++s) {
    const std::vector<float> scale_psfs(psfs.begin() + s * kFacets * kFreq * kPsfNpix,
                                        psfs.begin() + (s + 1) * kFacets * kFreq * kPsfNpix);
    const auto expected = host_gains(scale_psfs, weights, kFacets);
    for (int b = 0; b < kFacets; ++b)
      EXPECT_NEAR(all_gains.at(s * kFacets + b), expected.at(b), 1e-6f * expected.at(b))
          << "scale " << s << " facet " << b;
  }
}
