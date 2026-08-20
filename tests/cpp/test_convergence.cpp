#include <gtest/gtest.h>

#include <fast_deconv/common/convergence.hpp>
#include <limits>
#include <vector>

// Pure host tests — no GPU involved. These pin the CURRENT behavior of the
// convergence watchers, including the known quirk flagged by the TODO(guards)
// comment in convergence.cpp (last_rms_ shared across scales). When that is
// fixed, the corresponding tests below are the reviewable behavior change.

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

TEST(Convergence, FreshInstanceIsRunning)
{
  convergence c(/*max_iter=*/100, /*min_flux=*/0.1f, /*max_divergent_iter=*/3, /*divergence_factor=*/4.0f, no_stalls());
  EXPECT_EQ(c.status(), convergence_status::running);
  EXPECT_EQ(c.iteration(), 0);
  EXPECT_FALSE(c.should_stop());
}

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

TEST(Convergence, FluxDroppingBelowThresholdConverges)
{
  convergence c(100, 0.1f, 3, 4.0f, no_stalls());
  c.init(1.0f, 1.0f);
  c.track(0.5f, 1.0f, 1, 0);
  EXPECT_EQ(c.status(), convergence_status::running);
  c.track(0.05f, 1.0f, 1, 0);
  EXPECT_EQ(c.status(), convergence_status::converged);
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

// Divergence is on |flux|: a large negative flux relative to the previous one
// counts as a strike too.
TEST(Convergence, DivergenceUsesAbsoluteFlux)
{
  convergence c(100, -100.0f, /*max_divergent_iter=*/0, /*divergence_factor=*/2.0f, no_stalls());
  c.init(1.0f, 1.0f);
  c.track(-5.0f, 1.0f, 1, 0);  // | -5 | > 2*|1|: strike 1 > 0
  EXPECT_EQ(c.status(), convergence_status::diverged);
}

// update_status_ writes its checks weakest-reason first, so when several hold at
// once the LAST write wins. Pin the orderings that matter to the driver.
TEST(Convergence, StatusPrecedenceDivergedBeatsConvergedBeatsMaxIterations)
{
  // All three conditions trigger on the same update: diverged is reported.
  convergence c(/*max_iter=*/2, /*min_flux=*/-5.0f, /*max_divergent_iter=*/0, /*divergence_factor=*/1.5f, no_stalls());
  c.init(4.0f, 1.0f);
  c.track(-10.0f, 1.0f, 2, 0);  // iteration 2>=2, -10 <= -5, |−10| > 1.5*4
  EXPECT_EQ(c.status(), convergence_status::diverged);

  // Converged and max_iterations together: converged is reported.
  convergence c2(/*max_iter=*/2, /*min_flux=*/0.1f, /*max_divergent_iter=*/10, /*divergence_factor=*/100.0f,
                 no_stalls());
  c2.init(1.0f, 1.0f);
  c2.track(0.05f, 1.0f, 2, 0);
  EXPECT_EQ(c2.status(), convergence_status::converged);
}

// Slow exponential growth (~1%/iteration) never exceeds divergence_factor * previous
// flux. The initial-flux comparison catches it: past 4x the starting flux every
// iteration scores a strike, so it trips within max_divergent_iter + 1 of crossing.
TEST(Convergence, SlowExponentialDivergenceIsCaught)
{
  convergence c(/*max_iter=*/1000, /*min_flux=*/0.0f, /*max_divergent_iter=*/3, /*divergence_factor=*/4.0f,
                no_stalls());
  float flux = 1.0f;
  c.init(flux, 1.0f);
  int i = 0;
  for (; i < 200 && c.status() != convergence_status::diverged; ++i) {
    flux *= 1.01f;
    c.track(flux, 1.0f, 1, 0);
  }
  EXPECT_EQ(c.status(), convergence_status::diverged);
  // 1.01^139 first exceeds 4x, then 4 more iterations to exhaust the strike budget.
  EXPECT_EQ(i, 143);
}

// The initial-flux comparison must not fire on growth that stays under the factor.
TEST(Convergence, GrowthBelowDivergenceFactorIsTolerated)
{
  convergence c(/*max_iter=*/1000, /*min_flux=*/0.0f, /*max_divergent_iter=*/3, /*divergence_factor=*/4.0f,
                no_stalls());
  float flux = 1.0f;
  c.init(flux, 1.0f);
  for (int i = 0; i < 100; ++i) {  // 1.01^100 = 2.7 < 4
    flux *= 1.01f;
    c.track(flux, 1.0f, 1, 0);
  }
  EXPECT_EQ(c.status(), convergence_status::running);
}

// Once the residual overflows, every ordering test in update_status_ is false on NaN.
// Without the isfinite latch the status stays running and the run grinds to
// max_iteration, feeding a garbage residual to the next major cycle.
TEST(Convergence, NonFiniteFluxIsReportedAsDiverged)
{
  constexpr float nan_f = std::numeric_limits<float>::quiet_NaN();
  constexpr float inf_f = std::numeric_limits<float>::infinity();

  convergence c(100, 0.1f, /*max_divergent_iter=*/3, 4.0f, no_stalls());
  c.init(1.0f, 1.0f);
  c.track(nan_f, 1.0f, 1, 0);
  EXPECT_EQ(c.status(), convergence_status::diverged);
  EXPECT_TRUE(c.should_stop());

  // inf scores only one strike (3 are tolerated): the latch is what stops it.
  convergence c2(100, 0.1f, /*max_divergent_iter=*/3, 4.0f, no_stalls());
  c2.init(1.0f, 1.0f);
  c2.track(inf_f, 1.0f, 1, 0);
  EXPECT_EQ(c2.status(), convergence_status::diverged);

  // A non-finite INITIAL flux must not read as converged (NaN <= min_flux is false,
  // but so is every other test, which is how a NaN cycle reported success).
  convergence c3(100, 0.1f, 3, 4.0f, no_stalls());
  c3.init(nan_f, 1.0f);
  EXPECT_EQ(c3.status(), convergence_status::diverged);
}

// The rms can go non-finite while the peak is still a (huge) finite value scoring
// only one strike, so the flux alone leaves the run reading as running.
TEST(Convergence, NonFiniteRmsWithAFiniteFluxIsReportedAsDiverged)
{
  constexpr float inf_f = std::numeric_limits<float>::infinity();

  // max_iter is well above the 250 subminor iterations so the budget does not end the
  // run before the flux/rms checks are reached.
  convergence c(1000, 0.1f, /*max_divergent_iter=*/5, /*divergence_factor=*/1.3f, no_stalls());
  c.init(0.26f, 0.01f);
  c.track(3.36e21f, 1.0f, 250, 0);
  EXPECT_EQ(c.status(), convergence_status::running);

  convergence c2(1000, 0.1f, /*max_divergent_iter=*/5, /*divergence_factor=*/1.3f, no_stalls());
  c2.init(0.26f, 0.01f);
  c2.track(3.36e21f, inf_f, 250, 0);
  EXPECT_EQ(c2.status(), convergence_status::diverged);
  EXPECT_TRUE(c2.should_stop());

  // A non-finite INITIAL rms must not read as converged either.
  convergence c3(100, 0.1f, 5, 1.3f, no_stalls());
  c3.init(0.05f, inf_f);
  EXPECT_EQ(c3.status(), convergence_status::diverged);
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
  EXPECT_TRUE(c.is_stall(0));
  EXPECT_EQ(c.get_all_stalled(), std::vector<int>{0});
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

// Reaching the flux threshold is the good outcome and outranks both "ran out of work"
// statuses; a pathological residual outranks everything.
TEST(Convergence, StatusPrecedenceAgainstStalledAndNoComponents)
{
  convergence c(100, /*min_flux=*/0.1f, 10, 100.0f,
                scale_stall_tracker{/*n_scales=*/1, /*max_stall_count=*/0, /*stall_threshold=*/0.01f});
  c.init(1.0f, 5.0f);
  c.track(0.05f, 5.0f, 0, 0);  // stalled AND no components AND flux below threshold
  EXPECT_EQ(c.status(), convergence_status::converged);

  convergence c2(100, 0.1f, 10, 100.0f, no_stalls());
  c2.init(1.0f, 1.0f);
  c2.track(std::numeric_limits<float>::quiet_NaN(), 1.0f, 0, 0);
  EXPECT_EQ(c2.status(), convergence_status::diverged);
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
