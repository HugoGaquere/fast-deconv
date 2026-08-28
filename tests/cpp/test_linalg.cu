#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <random>
#include <vector>

#include "helpers/device_buffers.hpp"
#include "helpers/gpu_test.hpp"
#include "helpers/host_oracles.hpp"
#include "helpers/rng.hpp"

namespace core = fast_deconv::core;
namespace linalg = fast_deconv::linalg;
namespace fdtest = fast_deconv::test;

class WeightedSum : public fdtest::GpuTest {};

// out[i] = sum_f w[f] * A[f, i]. Weights deliberately do NOT sum to 1, so an
// accidental normalization inside the kernel would show up.
TEST_F(WeightedSum, RawPointerOverloadMatchesHostOracle)
{
  const int w = 3, n = 35;
  std::mt19937 rng(11);
  std::vector<float> a(w * n);
  fdtest::fill_uniform(rng, a, -1.0f, 1.0f);
  const std::vector<float> weights = {0.2f, 0.5f, 1.3f};

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_a(res(), sr, a);
  fdtest::device_buffer<float> d_w(res(), sr, weights);
  fdtest::device_buffer<float> d_out(res(), sr, static_cast<std::size_t>(n));

  linalg::weighted_sum_async(sr, d_a.get(), d_w.get(), d_out.get(), w, n);
  sr.wait();

  const auto out = d_out.to_host();
  const auto expected = fdtest::weighted_sum(a, weights, n);
  for (int i = 0; i < n; ++i) EXPECT_NEAR(out.at(i), expected.at(i), 1e-6f) << "pixel " << i;
}

TEST_F(WeightedSum, MdspanOverloadOnNonSquareImage)
{
  const int n_freq = 3, nrow = 7, ncol = 5;
  const int npix = nrow * ncol;
  std::mt19937 rng(23);
  std::vector<float> a(n_freq * npix);
  fdtest::fill_uniform(rng, a, -2.0f, 2.0f);
  const std::vector<float> weights = {0.7f, 0.1f, 0.6f};

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_a(res(), sr, a);
  fdtest::device_buffer<float> d_w(res(), sr, weights);
  fdtest::device_buffer<float> d_out(res(), sr, static_cast<std::size_t>(npix));

  core::device_span3d<float> a_view(d_a.get(), n_freq, nrow, ncol);
  core::span1d<float> w_view(d_w.get(), n_freq);
  core::device_span2d<float> out_view(d_out.get(), nrow, ncol);

  linalg::weighted_sum_async(sr, a_view, w_view, out_view);
  sr.wait();

  const auto out = d_out.to_host();
  const auto expected = fdtest::weighted_sum(a, weights, npix);
  for (int i = 0; i < npix; ++i) EXPECT_NEAR(out.at(i), expected.at(i), 1e-6f) << "pixel " << i;
}
