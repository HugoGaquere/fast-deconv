#include <gtest/gtest.h>

#include <fast_deconv/common/convergence.hpp>
#include <limits>
#include <vector>

// Pure host tests — no GPU involved. These pin the behavior of the convergence
// watchers: the flux/iteration/divergence paths, and the scale stall tracker's
// consecutive-strike counting.

namespace common = fast_deconv::common;
using common::convergence;
using common::convergence_status;
using common::scale_stall_tracker;

namespace {

// A stall threshold of 0 makes |Δrms| < threshold impossible, so this tracker never
// strikes and the tests below exercise the flux/iteration paths in isolation.
scale_stall_tracker no_stalls() { return scale_stall_tracker{/*n_scales=*/1, /*max_stall_count=*/0, 0.0f}; }

}  // namespace

// ============================================================================
// convergence
// ============================================================================

// init runs the status update but does not consume iterations, so an
// already-converged start stops at once.
TEST(Convergence, InitialFluxAtOrBelowThresholdConvergesImmediately)
{
  convergence c(100, 0.1f, 3, 4.0f, no_stalls());
  c.init(0.05f, 1.0f);
  EXPECT_EQ(c.status(), convergence_status::converged);
  EXPECT_TRUE(c.should_stop());
  EXPECT_EQ(c.iteration(), 0);
}

TEST(Convergence, IterationAccumulatesSubminorCountUpToMaxIterations)
{
  convergence c(/*max_iter=*/10, /*min_flux=*/0.0f, 3, 4.0f, no_stalls());
  c.init(1.0f, 1.0f);
  c.track(0.9f, 1.0f, /*subminor_count=*/4, /*selected_scale=*/0);
  EXPECT_EQ(c.iteration(), 4);
  EXPECT_EQ(c.status(), convergence_status::running);

  c.track(0.8f, 1.0f, 6, 0);
  EXPECT_EQ(c.iteration(), 10);
  EXPECT_EQ(c.status(), convergence_status::max_iterations);
  EXPECT_TRUE(c.should_stop());
}

// Divergence strikes are cumulative and never reset (DDFacet behavior): spikes
// separated by well-behaved iterations still add up to diverged.
TEST(Convergence, DivergentStrikesAccumulateAcrossCalmIterations)
{
  convergence c(100, 0.0f, /*max_divergent_iter=*/2, /*divergence_factor=*/2.0f, no_stalls());
  c.init(1.0f, 1.0f);

  c.track(3.0f, 1.0f, 1, 0);  // 3 > 2*1: strike 1
  c.track(1.0f, 1.0f, 1, 0);  // calm
  c.track(3.0f, 1.0f, 1, 0);  // strike 2
  c.track(1.0f, 1.0f, 1, 0);  // calm
  EXPECT_EQ(c.status(), convergence_status::running);

  c.track(3.0f, 1.0f, 1, 0);  // strike 3 > max_divergent_iter=2
  EXPECT_EQ(c.status(), convergence_status::diverged);
}

// update_status_ writes its checks weakest-reason first, so when several hold at
// once the LAST write wins. Pin every ordering the driver depends on: a pathological
// residual outranks everything, then converged, then the "ran out of work" statuses.
TEST(Convergence, StatusPrecedenceWhenSeveralReasonsHoldAtOnce)
{
  // diverged over converged over max_iterations: iteration 2>=2, -10 <= -5, |-10| > 1.5*4.
  convergence c(/*max_iter=*/2, /*min_flux=*/-5.0f, /*max_divergent_iter=*/0, /*divergence_factor=*/1.5f, no_stalls());
  c.init(4.0f, 1.0f);
  c.track(-10.0f, 1.0f, 2, 0);
  EXPECT_EQ(c.status(), convergence_status::diverged);

  // converged over max_iterations.
  convergence c2(/*max_iter=*/2, /*min_flux=*/0.1f, /*max_divergent_iter=*/10, /*divergence_factor=*/100.0f,
                 no_stalls());
  c2.init(1.0f, 1.0f);
  c2.track(0.05f, 1.0f, 2, 0);
  EXPECT_EQ(c2.status(), convergence_status::converged);

  // converged over all_scales_stalled and no_components.
  convergence c3(100, /*min_flux=*/0.1f, 10, 100.0f,
                 scale_stall_tracker{/*n_scales=*/1, /*max_stall_count=*/0, /*stall_threshold=*/0.01f});
  c3.init(1.0f, 5.0f);
  c3.track(0.05f, 5.0f, 0, 0);
  EXPECT_EQ(c3.status(), convergence_status::converged);
}

// Slow exponential growth (~1%/iteration) never exceeds divergence_factor * previous
// flux, so only the initial-flux comparison can catch it: below 4x the starting flux
// nothing is scored, past it every iteration earns a strike.
TEST(Convergence, SlowExponentialDivergenceIsCaughtOnlyPastTheFactor)
{
  convergence c(/*max_iter=*/1000, /*min_flux=*/0.0f, /*max_divergent_iter=*/3, /*divergence_factor=*/4.0f,
                no_stalls());
  float flux = 1.0f;
  c.init(flux, 1.0f);

  for (int i = 0; i < 100; ++i) {  // 1.01^100 = 2.7 < 4: tolerated
    flux *= 1.01f;
    c.track(flux, 1.0f, 1, 0);
  }
  EXPECT_EQ(c.status(), convergence_status::running);

  for (int i = 0; i < 100 && c.status() != convergence_status::diverged; ++i) {
    flux *= 1.01f;
    c.track(flux, 1.0f, 1, 0);
  }
  EXPECT_EQ(c.status(), convergence_status::diverged);
}

// Once the residual overflows, every ordering test in update_status_ is false on NaN. Without the isfinite
// latch the status stays running and the run grinds to max_iteration, feeding a garbage residual to the next
// major cycle.
TEST(Convergence, NonFiniteFluxOrRmsIsReportedAsDiverged)
{
  // NaN flux, in a cycle that also produced no component: the latch must outrank no_components.
  convergence c(100, 0.1f, /*max_divergent_iter=*/3, 4.0f, no_stalls());
  c.init(1.0f, 1.0f);
  c.track(std::numeric_limits<float>::quiet_NaN(), 1.0f, /*subminor_count=*/0, 0);
  EXPECT_EQ(c.status(), convergence_status::diverged);
  EXPECT_TRUE(c.should_stop());

  // The rms can go non-finite while the peak is still a (huge) finite value scoring only one of the five
  // tolerated strikes, so only the rms half of the latch catches it. max_iter is well above the 250
  // subminor iterations so the budget does not end the run first.
  convergence c2(1000, 0.1f, /*max_divergent_iter=*/5, /*divergence_factor=*/1.3f, no_stalls());
  c2.init(0.26f, 0.01f);
  c2.track(3.36e21f, std::numeric_limits<float>::infinity(), 250, 0);
  EXPECT_EQ(c2.status(), convergence_status::diverged);
}

// The owned stall tracker is seeded by init and updated by track, so retiring every
// scale ends the run through the same status the driver already polls.
TEST(Convergence, AllScalesStalledStopsTheRun)
{
  convergence c(/*max_iter=*/100, /*min_flux=*/0.0f, /*max_divergent_iter=*/10, /*divergence_factor=*/100.0f,
                scale_stall_tracker{/*n_scales=*/1, /*max_stall_count=*/0, /*stall_threshold=*/0.01f});
  c.init(1.0f, 5.0f);  // seeds last_rms_ = 5.0
  EXPECT_FALSE(c.should_stop());

  c.track(0.9f, 5.0f, 3, 0);  // |Δrms| = 0 < 0.01: strike 1 > 0, and scale 0 is the only scale
  EXPECT_EQ(c.status(), convergence_status::all_scales_stalled);
  EXPECT_TRUE(c.should_stop());
}

// An outer cycle that produced no component reports it instead of the driver breaking out.
TEST(Convergence, ZeroSubminorCountIsReportedAsNoComponents)
{
  convergence c(100, 0.0f, 10, 100.0f, no_stalls());
  c.init(1.0f, 1.0f);
  c.track(1.0f, 1.0f, /*subminor_count=*/0, /*selected_scale=*/0);
  EXPECT_EQ(c.status(), convergence_status::no_components);
  EXPECT_TRUE(c.should_stop());
  EXPECT_EQ(c.iteration(), 0);
}

// ============================================================================
// scale_stall_tracker
// ============================================================================

// Strikes are consecutive: real progress clears the count, so a scale that stalled in
// an earlier flux regime is not retired for it once it starts moving the rms again.
TEST(ScaleStallTracker, ProgressClearsEarlierStrikes)
{
  scale_stall_tracker t(/*n_scales=*/1, /*max_stall_count=*/1, /*stall_threshold=*/0.01f);
  t.init_rms(1.0f);

  t.update(0, 0.5f);  // big drop: no strike
  t.update(0, 0.5f);  // plateau: strike 1
  t.update(0, 0.9f);  // big change: no strike, and strike 1 IS reset
  EXPECT_FALSE(t.is_stall(0));

  t.update(0, 0.9f);  // plateau: strike 1 again, not 2
  EXPECT_FALSE(t.is_stall(0));

  t.update(0, 0.9f);  // plateau: strike 2 > 1
  EXPECT_TRUE(t.is_stall(0));
}

TEST(ScaleStallTracker, AllStalledTracksEveryScale)
{
  scale_stall_tracker t(/*n_scales=*/2, /*max_stall_count=*/0, /*stall_threshold=*/0.01f);
  EXPECT_FALSE(t.all_stalled());
  EXPECT_TRUE(t.get_all_stalled().empty());

  t.init_rms(1.0f);
  t.update(0, 1.0f);  // scale 0 stalls (count 1 > 0)
  EXPECT_TRUE(t.is_stall(0));
  EXPECT_FALSE(t.all_stalled());
  EXPECT_EQ(t.get_all_stalled(), std::vector<int>{0});

  t.update(1, 1.0f);  // scale 1 stalls
  EXPECT_TRUE(t.all_stalled());
  EXPECT_EQ(t.get_all_stalled(), (std::vector<int>{0, 1}));
}

