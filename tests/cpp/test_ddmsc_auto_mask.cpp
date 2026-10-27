#include <gtest/gtest.h>

#include <cmath>
#include <fast_deconv/algorithm/ddmsc.hpp>
#include <fast_deconv/common/convergence.hpp>
#include <fast_deconv/core/memory_types.hpp>
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

}  // namespace

class DdmscAutoMask : public fdtest::BackendTest {};

// Two point sources, the auto-mask forced on from the start and built from an earlier component on A only: B lies
// outside every scale's mask, so no component can reach it. The run must still converge once A is cleaned, rather
// than be held above the stop flux by B's untouchable peak.
TEST_F(DdmscAutoMask, PeakOutsideTheMaskDoesNotBlockConvergence)
{
  constexpr int kArow = 40, kAcol = 40, kBrow = 90, kBcol = 90;
  constexpr float kBflux = 5.0f;

  std::vector<float> h_plane(kNpix, 0.0f);
  add_point_source(h_plane, kArow, kAcol, 10.0);
  add_point_source(h_plane, kBrow, kBcol, kBflux);
  std::vector<float> h_dirty;
  for (int f = 0; f < kFreq; ++f) h_dirty.insert(h_dirty.end(), h_plane.begin(), h_plane.end());

  std::vector<float> h_psf_plane(kNpix, 0.0f);
  add_point_source(h_psf_plane, kN / 2, kN / 2, 1.0);
  std::vector<float> h_psfs;
  for (int f = 0; f < kFreq; ++f) h_psfs.insert(h_psfs.end(), h_psf_plane.begin(), h_psf_plane.end());

  const std::vector<float> h_xdes(kFreq, 1.0f);  // flat spectrum, one term
  const std::vector<float> h_jones(static_cast<std::size_t>(kFreq) * kNpix, 1.0f);
  const std::vector<float> h_weights(kFreq, 1.0f / kFreq);
  const auto h_mask = std::make_unique<bool[]>(kNpix);
  const std::vector<float> h_sigmas = {0.0f};
  const std::vector<float> h_bias = {1.0f};
  const std::vector<int> h_map(kNpix, 0);

  ddmsc::Ddmsc session(
      core::host_span4d<const float>(h_psfs.data(), 1, kFreq, kN, kN),
      core::host_span2d<const float>(h_xdes.data(), kFreq, 1), core::host_span2d<const bool>(h_mask.get(), kN, kN),
      core::host_span1d<const float>(h_sigmas.data(), 1), core::host_span1d<const float>(h_bias.data(), 1),
      core::host_span2d<const int>(h_map.data(), kN, kN), kN, kN);
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

  const auto result = session.run(core::host_span3d<float>(h_dirty.data(), kFreq, kN, kN),
                                  core::host_span3d<const float>(h_jones.data(), kFreq, kN, kN),
                                  core::host_span1d<const float>(h_weights.data(), kFreq));

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
