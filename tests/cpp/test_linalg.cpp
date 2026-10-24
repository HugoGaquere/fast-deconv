#include <gtest/gtest.h>

#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <vector>

#include "helpers/backend_test.hpp"

namespace core = fast_deconv::core;
namespace linalg = fast_deconv::linalg;
namespace fdtest = fast_deconv::test;

class WeightedSum : public fdtest::BackendTest {};

// out[i] = sum_f w[f] * a[f, i]. With a[f, i] = (f + 1) * (i + 1), out[i] = (i + 1) * sum_f w[f] * (f + 1):
// distinct per pixel and per channel, so a layout mix-up shows. The weights do not sum to 1, so a hidden
// normalization shows too.
TEST_F(WeightedSum, SumsTheChannelsWithTheirWeights)
{
  const int n_freq = 3, nrow = 7, ncol = 5, npix = nrow * ncol;
  const std::vector<float> weights = {0.7f, 0.1f, 0.6f};  // sum_f w[f] * (f + 1) = 0.7 + 0.2 + 1.8 = 2.7
  std::vector<float> a(n_freq * npix);
  for (int f = 0; f < n_freq; ++f)
    for (int i = 0; i < npix; ++i) a.at(f * npix + i) = static_cast<float>((f + 1) * (i + 1));
  std::vector<float> out(npix);

  const auto sr = res().make_ctx();
  const auto d_a = sr.stage(core::host_span3d<const float>(a.data(), n_freq, nrow, ncol));
  const auto d_weights = sr.stage(core::host_span1d<const float>(weights.data(), n_freq));
  auto d_out = sr.stage(core::host_span2d<float>(out.data(), nrow, ncol));

  linalg::weighted_sum(sr, d_a, d_weights, d_out);
  sr.unstage(d_out, out.data());
  sr.wait();

  for (int i = 0; i < npix; ++i) EXPECT_NEAR(out.at(i), 2.7f * (i + 1), 1e-4f) << "pixel " << i;
}
