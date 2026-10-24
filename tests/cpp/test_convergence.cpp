#include <gtest/gtest.h>

#include <fast_deconv/common/convergence.hpp>
#include <limits>
#include <vector>

// Pure host tests of the stop conditions of the minor cycle.

namespace common = fast_deconv::common;
using common::convergence;
using common::convergence_status;
using common::scale_stall_tracker;

namespace {

// A stall threshold of 0 makes |Δrms| < threshold impossible: this tracker never strikes.
scale_stall_tracker no_stalls() { return {/*n_scales=*/1, /*max_stall_count=*/0, /*stall_threshold=*/0.0f}; }

// Defaults keep every stop reason out of the way; each test sets the one it exercises.
struct config {
  int max_iter = 1000;
  float min_flux = 0.0f;
  int max_divergent_iter = 1000;
  float divergence_factor = 1000.0f;
  scale_stall_tracker stalls = no_stalls();
};

convergence make(const config& c)
{
  return convergence(c.max_iter, c.min_flux, c.max_divergent_iter, c.divergence_factor, c.stalls);
}

}  // namespace

// ============================================================================
// convergence
// ============================================================================

// An image already below the threshold stops at init, without using any iteration.
TEST(Convergence, InitialFluxBelowThresholdConvergesAtOnce)
{
  auto c = make({.min_flux = 0.1f});
  c.init(0.05f, 1.0f);
  EXPECT_EQ(c.status(), convergence_status::converged);
  EXPECT_EQ(c.iteration(), 0);
}

// Each track adds the cycle's component count; reaching max_iter stops the run.
TEST(Convergence, IterationsAddUpToTheBudget)
{
  auto c = make({.max_iter = 10});
  c.init(1.0f, 1.0f);
  c.track(0.9f, 1.0f, /*subminor_count=*/4, /*selected_scale=*/0);
  EXPECT_EQ(c.iteration(), 4);
  EXPECT_EQ(c.status(), convergence_status::running);

  c.track(0.8f, 1.0f, 6, 0);
  EXPECT_EQ(c.iteration(), 10);
  EXPECT_EQ(c.status(), convergence_status::max_iterations);
}

// A strike is a flux above divergence_factor * the previous one. Strikes add up and never reset (as in
// DDFacet), so spikes separated by calm iterations still end the run.
TEST(Convergence, DivergentStrikesAccumulateAcrossCalmIterations)
{
  auto c = make({.max_divergent_iter = 2, .divergence_factor = 2.0f});
  c.init(1.0f, 1.0f);
  c.track(3.0f, 1.0f, 1, 0);  // strike 1
  c.track(1.0f, 1.0f, 1, 0);
  c.track(3.0f, 1.0f, 1, 0);  // strike 2
  c.track(1.0f, 1.0f, 1, 0);
  EXPECT_EQ(c.status(), convergence_status::running);

  c.track(3.0f, 1.0f, 1, 0);  // strike 3 > 2
  EXPECT_EQ(c.status(), convergence_status::diverged);
}

// 1% growth per iteration never beats 4x the previous flux; only the comparison with the initial flux
// catches it, once the flux has grown past 4x where it started.
TEST(Convergence, SlowGrowthDivergesOncePastTheFactorOverTheInitialFlux)
{
  auto c = make({.max_divergent_iter = 3, .divergence_factor = 4.0f});
  float flux = 1.0f;
  c.init(flux, 1.0f);

  const auto grow = [&] {
    flux *= 1.01f;
    c.track(flux, 1.0f, 1, 0);
  };

  for (int i = 0; i < 100; ++i) grow();  // 1.01^100 = 2.7 < 4
  EXPECT_EQ(c.status(), convergence_status::running);

  for (int i = 0; i < 100 && !c.should_stop(); ++i) grow();  // 1.01^140 = 4.0, then 3 more strikes
  EXPECT_EQ(c.status(), convergence_status::diverged);
}

// When several reasons hold at once, reaching the flux threshold wins over running out of iterations or of
// scales: the run did converge.
TEST(Convergence, ConvergedWinsOverRunningOutOfWork)
{
  auto budget = make({.max_iter = 2, .min_flux = 0.1f});
  budget.init(1.0f, 1.0f);
  budget.track(0.05f, 1.0f, /*subminor_count=*/2, 0);
  EXPECT_EQ(budget.status(), convergence_status::converged);

  auto stalled = make({.min_flux = 0.1f, .stalls = {/*n_scales=*/1, /*max_stall_count=*/0, 0.01f}});
  stalled.init(1.0f, 5.0f);
  stalled.track(0.05f, 5.0f, 1, 0);  // flat rms: the only scale stalls
  EXPECT_EQ(stalled.status(), convergence_status::converged);
}

// Every comparison is false on NaN, so without an explicit check an overflowed residual would read as
// running and grind on to max_iter.
TEST(Convergence, NonFiniteFluxOrRmsIsReportedAsDiverged)
{
  // A NaN residual yields no component either; diverged must still win over no_components.
  auto c = make({});
  c.init(1.0f, 1.0f);
  c.track(std::numeric_limits<float>::quiet_NaN(), 1.0f, /*subminor_count=*/0, 0);
  EXPECT_EQ(c.status(), convergence_status::diverged);

  // From a real run: the rms overflowed while the peak was still finite and scored a single strike.
  auto c2 = make({.max_divergent_iter = 5, .divergence_factor = 1.3f});
  c2.init(0.26f, 0.01f);
  c2.track(3.36e21f, std::numeric_limits<float>::infinity(), 250, 0);
  EXPECT_EQ(c2.status(), convergence_status::diverged);
}

TEST(Convergence, AllScalesStalledStopsTheRun)
{
  auto c = make({.stalls = {/*n_scales=*/1, /*max_stall_count=*/0, /*stall_threshold=*/0.01f}});
  c.init(1.0f, 5.0f);
  c.track(0.9f, 5.0f, 3, 0);  // |Δrms| = 0 < 0.01: one strike > 0 on the only scale
  EXPECT_EQ(c.status(), convergence_status::all_scales_stalled);
}

TEST(Convergence, CycleWithoutComponentsStopsTheRun)
{
  auto c = make({});
  c.init(1.0f, 1.0f);
  c.track(1.0f, 1.0f, /*subminor_count=*/0, 0);
  EXPECT_EQ(c.status(), convergence_status::no_components);
  EXPECT_EQ(c.iteration(), 0);
}

// ============================================================================
// scale_stall_tracker
// ============================================================================

// A strike is an rms change below the threshold; a scale stalls past max_stall_count consecutive strikes.
// Real progress resets the count, so a scale that plateaued once is not retired for it later.
TEST(ScaleStallTracker, StallsOnConsecutiveStrikesOnly)
{
  scale_stall_tracker t(/*n_scales=*/1, /*max_stall_count=*/1, /*stall_threshold=*/0.01f);
  t.init_rms(1.0f);

  t.update(0, 0.5f);  // progress
  t.update(0, 0.5f);  // strike 1
  t.update(0, 0.9f);  // progress: count reset
  t.update(0, 0.9f);  // strike 1 again
  EXPECT_FALSE(t.is_stall(0));

  t.update(0, 0.9f);  // strike 2 > 1
  EXPECT_TRUE(t.is_stall(0));
}

TEST(ScaleStallTracker, AllStalledOnlyOnceEveryScaleIs)
{
  scale_stall_tracker t(/*n_scales=*/2, /*max_stall_count=*/0, /*stall_threshold=*/0.01f);
  t.init_rms(1.0f);
  EXPECT_TRUE(t.get_all_stalled().empty());

  t.update(0, 1.0f);
  EXPECT_FALSE(t.all_stalled());
  EXPECT_EQ(t.get_all_stalled(), std::vector<int>{0});

  t.update(1, 1.0f);
  EXPECT_TRUE(t.all_stalled());
  EXPECT_EQ(t.get_all_stalled(), (std::vector<int>{0, 1}));
}
