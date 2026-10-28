#include <gtest/gtest.h>

#include <cmath>
#include <fast_deconv/algorithm/ddmsc.hpp>
#include <fast_deconv/common/convergence.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <initializer_list>
#include <memory>
#include <optional>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/host_oracles.hpp"

// Full DDMSC runs with the auto-mask engaged.

namespace common = fast_deconv::common;
namespace core = fast_deconv::core;
namespace ddmsc = fast_deconv::algorithm::ddmsc;
namespace fdtest = fast_deconv::test;

using fdtest::flat;

namespace {

constexpr int kN = 128;
constexpr int kNpix = kN * kN;
constexpr int kFreq = 2;
constexpr double kSigmaPsf = 2.0;

// Adds a Gaussian of peak amplitude, centered at (row, col).
void add_point_source(std::vector<float>& plane, int row, int col, double amplitude)
{
  for (int r = 0; r < kN; ++r)
    for (int c = 0; c < kN; ++c) {
      const double d2 = (r - row) * static_cast<double>(r - row) + (c - col) * static_cast<double>(c - col);
      plane.at(flat(r, c, kN)) += static_cast<float>(amplitude * std::exp(-d2 / (2.0 * kSigmaPsf * kSigmaPsf)));
    }
}

// Point sources on a flat-spectrum sky, one facet, one delta scale, seen through a Gaussian PSF.
struct scene {
  struct source {
    int row;
    int col;
    double amplitude;
  };

  std::vector<float> dirty;
  std::vector<float> psfs;
  std::vector<float> xdes = std::vector<float>(kFreq, 1.0f);  // flat spectrum, one term
  std::vector<float> jones = std::vector<float>(static_cast<std::size_t>(kFreq) * kNpix, 1.0f);
  std::vector<float> weights = std::vector<float>(kFreq, 1.0f / kFreq);
  std::unique_ptr<bool[]> mask = std::make_unique<bool[]>(kNpix);
  std::vector<float> sigmas = {0.0f};
  std::vector<float> bias = {1.0f};
  std::vector<int> map = std::vector<int>(kNpix, 0);

  explicit scene(std::initializer_list<source> sources)
  {
    std::vector<float> plane(kNpix, 0.0f);
    for (const auto& s : sources) add_point_source(plane, s.row, s.col, s.amplitude);
    std::vector<float> psf_plane(kNpix, 0.0f);
    add_point_source(psf_plane, kN / 2, kN / 2, 1.0);
    for (int f = 0; f < kFreq; ++f) {
      dirty.insert(dirty.end(), plane.begin(), plane.end());
      psfs.insert(psfs.end(), psf_plane.begin(), psf_plane.end());
    }
  }

  ddmsc::Ddmsc make_session() const
  {
    return {core::host_span4d<const float>(psfs.data(), 1, kFreq, kN, kN),
            core::host_span2d<const float>(xdes.data(), kFreq, 1),
            core::host_span2d<const bool>(mask.get(), kN, kN),
            core::host_span1d<const float>(sigmas.data(), 1),
            core::host_span1d<const float>(bias.data(), 1),
            core::host_span2d<const int>(map.data(), kN, kN),
            kN,
            kN};
  }

  ddmsc::ddmsc_result run(ddmsc::Ddmsc& session, std::vector<float>& residual) const
  {
    return session.run(core::host_span3d<float>(residual.data(), kFreq, kN, kN),
                       core::host_span3d<const float>(jones.data(), kFreq, kN, kN),
                       core::host_span1d<const float>(weights.data(), kFreq));
  }
};

}  // namespace

class DdmscAutoMask : public fdtest::BackendTest {};

// Two point sources, the auto-mask forced on from the start and built from an earlier component on A only: B lies
// outside every scale's mask, so no component can reach it. The run must still converge once A is cleaned, rather
// than be held above the stop flux by B's untouchable peak.
TEST_F(DdmscAutoMask, PeakOutsideTheMaskDoesNotBlockConvergence)
{
  constexpr int kArow = 40, kAcol = 40, kBrow = 90, kBcol = 90;
  constexpr float kBflux = 5.0f;

  scene sky({{kArow, kAcol, 10.0}, {kBrow, kBcol, kBflux}});
  auto& h_dirty = sky.dirty;
  auto session = sky.make_session();
  session.ctx.historical_peak_coords = {{kArow, kAcol}};
  session.ctx.historical_scales = {0};

  session.cfg = {
      .max_iteration = 2000,
      .divergence_factor = 10.0f,
      .flux_threshold = 1.0f,  // below B: reachable only if B is left out of the stop test
      .stop_rms_factor = 0.0f,
      .stop_peak_factor = 0.0f,
      .stop_cycle_factor = 0.0f,
      .stop_sidelobe_level = 0.0f,
      .clean_negative = false,
      .peak_factor = 0.15f,
      .gamma = 0.1f,
      .max_clean_iteration = 50,
      .scale_stall_threshold = 0.0f,
      .enable_auto_mask = true,
      .force_enable_auto_mask = true,
      .auto_mask_peak_threshold = std::nullopt,
      .auto_mask_rms_threshold = std::nullopt,
  };

  const auto result = sky.run(session, h_dirty);

  EXPECT_EQ(result.status, common::convergence_status::converged) << common::to_string(result.status);
  // The minor cycle stopped on the masked peak, but the caller deciding on another major cycle must see B.
  EXPECT_NEAR(result.final_flux, kBflux, 1e-3f) << "final_flux must be the unmasked peak";
  ASSERT_GT(result.n_components(), 0u);
  for (const auto& [r, c] : result.peak_coords) {
    EXPECT_LE(std::abs(r - kArow), 3) << "component at (" << r << ", " << c << ") is outside A's mask";
    EXPECT_LE(std::abs(c - kAcol), 3) << "component at (" << r << ", " << c << ") is outside A's mask";
  }
  EXPECT_NEAR(h_dirty.at(flat(kBrow, kBcol, kN)), kBflux, 1e-3f) << "B is masked out and must stay uncleaned";
}

// release() between two major cycles frees the whole pool; the next run() re-stages its inputs and builds the forced
// auto-mask from the kept history, exactly like a session that never released.
TEST_F(DdmscAutoMask, ReleaseFreesThePoolAndKeepsTheHistory)
{
  const scene sky({{40, 40, 10.0}, {90, 90, 5.0}});
  auto released = sky.make_session();
  auto kept = sky.make_session();
  std::vector<float> released_dirty = sky.dirty;
  std::vector<float> kept_dirty = sky.dirty;

  for (auto* s : {&released, &kept}) s->cfg = {.max_iteration = 20, .max_clean_iteration = 10};
  const auto first = sky.run(released, released_dirty);
  sky.run(kept, kept_dirty);
  ASSERT_GT(first.n_components(), 0u);

  released.release();
  EXPECT_EQ(released.ctx.resources.pool_used_bytes(), 0u);
  EXPECT_EQ(released.ctx.resources.pool_reserved_bytes(), 0u);
  EXPECT_EQ(released.ctx.historical_peak_coords.size(), first.n_components());

  for (auto* s : {&released, &kept}) {
    s->cfg.max_iteration = 200;
    s->cfg.enable_auto_mask = true;
    s->cfg.force_enable_auto_mask = true;
  }
  const auto after_release = sky.run(released, released_dirty);
  const auto reference = sky.run(kept, kept_dirty);

  ASSERT_GT(reference.n_components(), 0u);
  EXPECT_EQ(after_release.peak_coords, reference.peak_coords);
  EXPECT_EQ(after_release.scales, reference.scales);
  EXPECT_EQ(after_release.status, reference.status);
  for (std::size_t i = 0; i < kept_dirty.size(); ++i)
    ASSERT_NEAR(released_dirty[i], kept_dirty[i], 1e-5f) << "residual differs at " << i;
}
