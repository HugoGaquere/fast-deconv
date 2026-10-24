#include <gtest/gtest.h>

#include <fast_deconv/common/gain.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/host_oracles.hpp"

namespace core = fast_deconv::core;
namespace common = fast_deconv::common;
namespace fdtest = fast_deconv::test;

using fdtest::flat;

class ComputeGainBatched : public fdtest::BackendTest {};

// gain = gamma / peak of the channel-weighted mean PSF, one per facet.
TEST_F(ComputeGainBatched, IsGammaOverThePeakOfTheWeightedMeanPsf)
{
  constexpr int n_facets = 2, n_freq = 3, n = 9;
  const std::vector<float> weights = {0.5f, 0.3f, 0.2f};
  std::vector<float> psfs(n_facets * n_freq * n * n, 0.0f);
  const auto at = [&](int facet, int f, int r, int c) -> float& {
    return psfs.at((facet * n_freq + f) * n * n + flat(r, c, n));
  };

  // Facet 0: every channel peaks at 1 in the center, so the mean peaks at 0.5 + 0.3 + 0.2 = 1.
  for (int f = 0; f < n_freq; ++f) at(0, f, 4, 4) = 1.0f;
  // Facet 1: the mean is 0.5 * 1 at (2, 6) and 0.2 * 4 = 0.8 at (7, 7), so it peaks at 0.8.
  at(1, 0, 2, 6) = 1.0f;
  at(1, 2, 7, 7) = 4.0f;

  const auto sr = res().make_ctx();
  const auto d_psfs = sr.stage(core::host_span4d<const float>(psfs.data(), n_facets, n_freq, n, n));
  const auto d_weights = sr.stage(core::host_span1d<const float>(weights.data(), n_freq));

  const auto gains = common::compute_gain_batched(sr, d_psfs, d_weights, /*gamma=*/0.1f);

  ASSERT_EQ(gains.size(), 2u);
  EXPECT_FLOAT_EQ(gains.at(0), 0.1f / 1.0f);
  EXPECT_FLOAT_EQ(gains.at(1), 0.1f / 0.8f);
}
