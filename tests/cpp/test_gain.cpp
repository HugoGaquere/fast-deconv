#include <gtest/gtest.h>

#include <fast_deconv/common/gain.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <random>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/device_buffers.hpp"
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

class GainBatched : public fdtest::BackendTest {};

TEST_F(GainBatched, PerFacetGainMatchesWeightedMeanMaxOracle)
{
  std::mt19937 rng(89);
  std::vector<float> psfs(kFacets * kFreq * kPsfNpix);
  fdtest::fill_uniform(rng, psfs, 0.0f, 0.5f);
  // Distinct known peaks so the two facets produce clearly different gains.
  psfs.at(0 * kFreq * kPsfNpix + 0 * kPsfNpix + fdtest::flat(4, 4, kPsfW)) = 1.0f;
  psfs.at(1 * kFreq * kPsfNpix + 1 * kPsfNpix + fdtest::flat(2, 6, kPsfW)) = 2.0f;
  const std::vector<float> weights = {0.5f, 0.3f, 0.2f};

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_psfs(sr, psfs);
  fdtest::device_buffer<float> d_w(sr, weights);

  core::span4d<float> psf_view(d_psfs.get(), kFacets, kFreq, kPsfH, kPsfW);
  core::span1d<float> w_view(d_w.get(), kFreq);

  const auto gains = common::compute_gain_batched(sr, psf_view, w_view, kGamma);
  const auto expected = host_gains(psfs, weights, kFacets);

  ASSERT_EQ(gains.size(), static_cast<std::size_t>(kFacets));
  for (int b = 0; b < kFacets; ++b) EXPECT_NEAR(gains.at(b), expected.at(b), 1e-6f * expected.at(b)) << "facet " << b;
}
