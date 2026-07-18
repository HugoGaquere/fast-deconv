#include <gtest/gtest.h>

#include <fast_deconv/common/convergence.hpp>
#include <vector>

// Pure host tests — no GPU involved. These pin the CURRENT behavior of the
// convergence watchers, including two known quirks flagged by TODO(guards)
// comments in convergence.cpp (slow divergence uncaught; last_rms_ shared
// across scales). When those are fixed, the corresponding tests below are the
// reviewable behavior change.

namespace common = fast_deconv::common;
using common::convergence;
using common::convergence_status;
using common::scale_stall_tracker;

// ============================================================================
// convergence
// ============================================================================

TEST(Convergence, FreshInstanceIsNotConverged)
{
  convergence c(/*max_iter=*/100, /*min_flux=*/0.1f, /*max_divergent_iter=*/3, /*divergence_factor=*/4.0f);
  EXPECT_EQ(c.status(), convergence_status::not_converged);
  EXPECT_EQ(c.iteration(), 0);
  EXPECT_FALSE(c.should_stop());
}

// The first track_flux call is the initial flux: it runs the status update but
// does not consume iterations, so an already-converged start stops at once.
TEST(Convergence, InitialFluxAtOrBelowThresholdConvergesImmediately)
{
  convergence c(100, 0.1f, 3, 4.0f);
  c.track_flux(0.05f);
  EXPECT_EQ(c.status(), convergence_status::flux_converged);
  EXPECT_TRUE(c.should_stop());
  EXPECT_EQ(c.iteration(), 0);
}

TEST(Convergence, IterationAccumulatesSubminorCountUpToMaxIterations)
{
  convergence c(/*max_iter=*/10, /*min_flux=*/0.0f, 3, 4.0f);
  c.track_flux(1.0f);  // initial flux, no iterations consumed
  c.track_flux(0.9f, /*subminor_count=*/4);
  EXPECT_EQ(c.iteration(), 4);
  EXPECT_EQ(c.status(), convergence_status::not_converged);

  c.track_flux(0.8f, 6);
  EXPECT_EQ(c.iteration(), 10);
  EXPECT_EQ(c.status(), convergence_status::max_iterations);
  EXPECT_TRUE(c.should_stop());
}

TEST(Convergence, FluxDroppingBelowThresholdConverges)
{
  convergence c(100, 0.1f, 3, 4.0f);
  c.track_flux(1.0f);
  c.track_flux(0.5f);
  EXPECT_EQ(c.status(), convergence_status::not_converged);
  c.track_flux(0.05f);
  EXPECT_EQ(c.status(), convergence_status::flux_converged);
}

// Divergence strikes are cumulative and never reset (DDFacet behavior): spikes
// separated by well-behaved iterations still add up to flux_diverged.
TEST(Convergence, DivergentStrikesAccumulateAcrossCalmIterations)
{
  convergence c(100, 0.0f, /*max_divergent_iter=*/2, /*divergence_factor=*/2.0f);
  c.track_flux(1.0f);

  c.track_flux(3.0f);  // 3 > 2*1: strike 1
  c.track_flux(1.0f);  // calm
  c.track_flux(3.0f);  // strike 2
  c.track_flux(1.0f);  // calm
  EXPECT_EQ(c.status(), convergence_status::not_converged);

  c.track_flux(3.0f);  // strike 3 > max_divergent_iter=2
  EXPECT_EQ(c.status(), convergence_status::flux_diverged);
}

// Divergence is on |flux|: a large negative flux relative to the previous one
// counts as a strike too.
TEST(Convergence, DivergenceUsesAbsoluteFlux)
{
  convergence c(100, -100.0f, /*max_divergent_iter=*/0, /*divergence_factor=*/2.0f);
  c.track_flux(1.0f);
  c.track_flux(-5.0f);  // | -5 | > 2*|1|: strike 1 > 0
  EXPECT_EQ(c.status(), convergence_status::flux_diverged);
}

// update_status_ writes the checks in order max_iterations → converged →
// diverged, so when several hold at once the LAST write wins. Pin both
// orderings that matter to the driver.
TEST(Convergence, StatusPrecedenceDivergedBeatsConvergedBeatsMaxIterations)
{
  // All three conditions trigger on the same update: diverged is reported.
  convergence c(/*max_iter=*/2, /*min_flux=*/-5.0f, /*max_divergent_iter=*/0, /*divergence_factor=*/1.5f);
  c.track_flux(4.0f);
  c.track_flux(-10.0f, 2);  // iteration 2>=2, -10 <= -5, |−10| > 1.5*4
  EXPECT_EQ(c.status(), convergence_status::flux_diverged);

  // Converged and max_iterations together: converged is reported.
  convergence c2(/*max_iter=*/2, /*min_flux=*/0.1f, /*max_divergent_iter=*/10, /*divergence_factor=*/100.0f);
  c2.track_flux(1.0f);
  c2.track_flux(0.05f, 2);
  EXPECT_EQ(c2.status(), convergence_status::flux_converged);
}

// Pin the known gap flagged at convergence.cpp TODO(guards) (track_flux):
// slow exponential growth (~1%/iteration) never exceeds
// divergence_factor * previous flux, so it is never reported as divergence —
// the run only ends via max_iterations.
TEST(Convergence, SlowExponentialDivergenceIsNotCaught)
{
  convergence c(/*max_iter=*/50, /*min_flux=*/0.0f, /*max_divergent_iter=*/3, /*divergence_factor=*/4.0f);
  float flux = 1.0f;
  c.track_flux(flux);
  for (int i = 0; i < 49; ++i) {
    flux *= 1.01f;
    c.track_flux(flux);
    ASSERT_NE(c.status(), convergence_status::flux_diverged) << "at iteration " << i;
  }
  EXPECT_EQ(c.status(), convergence_status::not_converged);

  c.track_flux(flux * 1.01f);  // 50th iteration: stops, but only on the budget
  EXPECT_EQ(c.status(), convergence_status::max_iterations);
}

// ============================================================================
// scale_stall_tracker
// ============================================================================

TEST(ScaleStallTracker, StallsAfterRepeatedSmallRmsChanges)
{
  scale_stall_tracker t(/*n_scales=*/3, /*max_stall_count=*/2, /*stall_threshold=*/0.01f);
  t.init_rms(1.0f);

  t.update(0, 1.0f);            // |Δrms| = 0 < threshold: strike 1
  t.update(0, 1.0f);            // strike 2
  EXPECT_FALSE(t.is_stall(0));  // stall requires count > max_stall_count

  t.update(0, 1.0f);  // strike 3 > 2
  EXPECT_TRUE(t.is_stall(0));
  EXPECT_FALSE(t.is_stall(1));
  EXPECT_FALSE(t.is_stall(2));
}

// Strikes are cumulative and never reset (DDFacet behavior): real progress
// between plateaus does not clear previously earned strikes.
TEST(ScaleStallTracker, StrikesAccumulateAcrossRealProgress)
{
  scale_stall_tracker t(/*n_scales=*/1, /*max_stall_count=*/1, /*stall_threshold=*/0.01f);
  t.init_rms(1.0f);

  t.update(0, 0.5f);  // big drop: no strike
  t.update(0, 0.5f);  // plateau: strike 1
  t.update(0, 0.9f);  // big change: no strike, and strike 1 is NOT reset
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

// Pin the known quirk flagged at convergence.cpp TODO(guards) (update):
// last_rms_ is shared across scales, so a scale is judged against the rms left
// by whichever scale updated before it — scale 0's plateau below hands scale 1
// a strike even though scale 1 was never updated before.
TEST(ScaleStallTracker, SharedLastRmsLetsOneScaleStrikeAnother)
{
  scale_stall_tracker t(/*n_scales=*/2, /*max_stall_count=*/0, /*stall_threshold=*/0.01f);
  t.init_rms(1.0f);

  t.update(0, 1.0f);  // plateau on scale 0; last_rms_ stays 1.0
  t.update(1, 1.0f);  // compared against scale 0's rms: scale 1 gets a strike
  EXPECT_TRUE(t.is_stall(1));
}
