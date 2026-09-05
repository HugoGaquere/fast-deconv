#include <gtest/gtest.h>

#include <cstddef>
#include <fast_deconv/algorithm/conv_psf_cache.hpp>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <random>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/device_buffers.hpp"
#include "helpers/rng.hpp"

namespace core = fast_deconv::core;
namespace linalg = fast_deconv::linalg;
namespace scale = fast_deconv::scale;
namespace fdtest = fast_deconv::test;

using fast_deconv::algorithm::conv_psf_cache;

// ============================================================================
// conv_psf_cache — on-demand (scale, facet) entries under an LRU byte budget.
// The convolution itself is covered by test_scales; what matters here is that
// the cache hands back the same buffers, counts hits, and keeps an evicted
// entry alive for whoever already holds it.
// ============================================================================

class ConvPsfCacheTest : public fdtest::BackendTest {
 protected:
  static constexpr int kFacets = 3;
  static constexpr int kFreq = 2;
  static constexpr int kH = 32;
  static constexpr int kW = 32;
  static constexpr int kNpix = kH * kW;
  static constexpr float kGamma = 0.1f;

  ConvPsfCacheTest()
      : build_lane(res().make_ctx()),
        reader_lane(res().make_ctx()),
        conv(build_lane, kH, kW, /*forward_batch=*/kFreq, /*backward_batch=*/kFreq, /*n_backward_plans=*/2,
             /*padding=*/1.5f),
        work(build_lane, conv),
        d_psfs(build_lane, random_psfs()),
        d_sigmas(build_lane, std::vector<float>{0.0f, 1.0f, 2.0f}),
        d_weights(build_lane, std::vector<float>{0.6f, 0.4f}),
        cache(conv, reader_lane, psf_view(), sigma_view(), /*budget_bytes=*/0)
  {
    cache.configure(weight_view(), {0.6f, 0.4f}, kGamma);
  }

  static std::vector<float> random_psfs()
  {
    std::mt19937 rng(4242);
    std::vector<float> psfs(static_cast<std::size_t>(kFacets) * kFreq * kNpix);
    fdtest::fill_uniform(rng, psfs, 0.1f, 1.0f);
    return psfs;
  }

  core::span4d<float> psf_view() const { return core::span4d<float>(d_psfs.get(), kFacets, kFreq, kH, kW); }
  core::span1d<float> sigma_view() const { return core::span1d<float>(d_sigmas.get(), 3); }
  core::span1d<const float> weight_view() const { return core::span1d<const float>(d_weights.get(), kFreq); }

  std::vector<float> to_host(core::span3d<float> v)
  {
    std::vector<float> host(v.size());
    build_lane.download(v, host.data());
    build_lane.wait();
    return host;
  }

  core::exec_ctx build_lane;
  core::exec_ctx reader_lane;
  linalg::convolve_ctx conv;
  fdtest::scoped_work_area work;
  fdtest::device_buffer<float> d_psfs;
  fdtest::device_buffer<float> d_sigmas;
  fdtest::device_buffer<float> d_weights;
  conv_psf_cache cache;
};

// A second get() of the same pair must not convolve again, and must return the
// very same buffers.
TEST_F(ConvPsfCacheTest, SecondGetOfAPairIsAHit)
{
  const auto first = cache.get(1, 0);
  EXPECT_EQ(cache.get_stats().misses, 1);
  EXPECT_EQ(cache.get_stats().hits, 0);

  const auto second = cache.get(1, 0);
  EXPECT_EQ(cache.get_stats().misses, 1);
  EXPECT_EQ(cache.get_stats().hits, 1);
  EXPECT_EQ(first.conv.data_handle(), second.conv.data_handle());
  EXPECT_EQ(first.conv2.data_handle(), second.conv2.data_handle());
  EXPECT_EQ(first.gain, second.gain);
}

// Scale 0 is the delta kernel: conv_psf is the raw PSF, so the entry must point
// straight at it (no copy, no bytes charged) and take gamma as its gain.
TEST_F(ConvPsfCacheTest, ScaleZeroAliasesTheRawPsf)
{
  const auto e = cache.get(0, 1);
  EXPECT_EQ(e.conv.data_handle(), d_psfs.get() + static_cast<std::size_t>(1) * kFreq * kNpix);
  EXPECT_FLOAT_EQ(e.gain, kGamma);
  EXPECT_EQ(cache.entry_bytes(0), static_cast<std::size_t>(kNpix) * sizeof(float));
  EXPECT_EQ(cache.bytes_resident(), cache.entry_bytes(0));
}

// The per-facet build path must agree with the batched entry point it replaced.
TEST_F(ConvPsfCacheTest, MatchesTheBatchedConvolution)
{
  fdtest::device_buffer<float> d_conv(build_lane, static_cast<std::size_t>(kFacets) * kFreq * kNpix);
  fdtest::device_buffer<float> d_conv2(build_lane, static_cast<std::size_t>(kFacets) * kNpix);
  core::span1d<float> sigma_one(d_sigmas.get() + 1, 1);
  scale::convolve_psfs_with_scale_async(conv, psf_view(), sigma_one, /*scale_idx=*/1, weight_view(),
                                        core::span4d<float>(d_conv.get(), kFacets, kFreq, kH, kW),
                                        core::span3d<float>(d_conv2.get(), kFacets, kH, kW));
  build_lane.wait();
  const auto expected = d_conv.to_host();

  for (int f = 0; f < kFacets; ++f) {
    const auto got = to_host(cache.get(1, f).conv);
    for (std::size_t i = 0; i < got.size(); ++i)
      ASSERT_NEAR(got.at(i), expected.at(f * kFreq * kNpix + i), 1e-5f) << "facet " << f << " flat " << i;
  }
}

TEST_F(ConvPsfCacheTest, PrefetchScaleBuildsEveryFacetOnce)
{
  cache.prefetch_scale(1);
  EXPECT_EQ(cache.get_stats().misses, kFacets);
  EXPECT_EQ(cache.bytes_resident(), kFacets * cache.entry_bytes(1));

  cache.prefetch_scale(1);
  EXPECT_EQ(cache.get_stats().misses, kFacets);
  EXPECT_EQ(cache.get_stats().hits, kFacets);
}

// Over budget, the least recently used pair goes first — and a copy taken
// before the eviction still reads the right data afterwards.
TEST_F(ConvPsfCacheTest, BudgetEvictsLeastRecentlyUsed)
{
  cache.set_budget_bytes(2 * cache.entry_bytes(1));

  const auto first = cache.get(1, 0);
  const auto before_eviction = to_host(first.conv);
  cache.get(1, 1);
  EXPECT_EQ(cache.get_stats().evictions, 0);

  cache.get(1, 2);  // third entry does not fit: (1, 0) is the coldest
  EXPECT_EQ(cache.get_stats().evictions, 1);
  EXPECT_EQ(cache.bytes_resident(), 2 * cache.entry_bytes(1));
  EXPECT_EQ(to_host(first.conv), before_eviction);

  cache.get(1, 0);  // evicted, so it is rebuilt
  EXPECT_EQ(cache.get_stats().misses, 4);
}

// A budget below one entry would evict everything and still not fit, so the
// entry is handed over uncached instead of thrashing.
TEST_F(ConvPsfCacheTest, EntryLargerThanBudgetIsNotCached)
{
  cache.set_budget_bytes(cache.entry_bytes(1) / 2);
  cache.get(1, 0);
  cache.get(1, 0);
  EXPECT_EQ(cache.bytes_resident(), 0u);
  EXPECT_EQ(cache.get_stats().misses, 2);
}

// The entries are a function of the weights and gamma, so moving either drops them.
TEST_F(ConvPsfCacheTest, ConfigureDropsTheCacheWhenGammaMoves)
{
  const float gain = cache.get(1, 0).gain;
  EXPECT_GT(cache.bytes_resident(), 0u);

  cache.configure(weight_view(), {0.6f, 0.4f}, kGamma);  // unchanged: keeps the entry
  EXPECT_GT(cache.bytes_resident(), 0u);
  EXPECT_EQ(cache.get_stats().misses, 1);

  cache.configure(weight_view(), {0.6f, 0.4f}, 2 * kGamma);
  EXPECT_EQ(cache.bytes_resident(), 0u);
  // gain = gamma / max(weighted mean), so doubling gamma doubles it.
  EXPECT_NEAR(cache.get(1, 0).gain, 2 * gain, 1e-6f);
  EXPECT_EQ(cache.get_stats().misses, 2);
}
