#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fast_deconv/algorithm/ddmsc_cycles.hpp>
#include <fast_deconv/algorithm/ddmsc_types.hpp>
#include <fast_deconv/common/convergence.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/device_buffers.hpp"
#include "helpers/host_oracles.hpp"
#include "helpers/rng.hpp"

// ============================================================================
// Non-regression test: a full run_ddmsc_cycles pass on a fully synthetic,
// seeded scene. Scalar METRICS of the run (not arrays) are compared against
// the committed baseline tests/baselines/ddmsc_synthetic.json with loose,
// per-metric tolerances so the test survives GPU / cuFFT-version drift while
// still catching algorithmic regressions.
//
// Regenerate the baseline after an intentional change:
//   FAST_DECONV_UPDATE_BASELINE=1 ctest --test-dir <build> -L NONREG
// then review and commit the JSON diff.
// ============================================================================

namespace common = fast_deconv::common;
namespace core = fast_deconv::core;
namespace ddmsc = fast_deconv::algorithm::ddmsc;
namespace fdtest = fast_deconv::test;

using json = nlohmann::ordered_json;

namespace {

// ---- Scene definition (any change here requires regenerating the baseline;
// ---- the config block in the JSON guards against a stale one).
constexpr int kSeed = 42;
constexpr int kNrow = 256;
constexpr int kNcol = 256;
constexpr int kNpix = kNrow * kNcol;
constexpr int kFreq = 4;
constexpr int kScales = 4;
constexpr int kOrder = 2;
constexpr int kFacets = 1;
constexpr double kSigmaPsf = 2.0;     // unit-peak Gaussian PSF width (px)
constexpr float kNoiseSigma = 0.01f;  // per-channel noise std

constexpr std::array<double, kFreq> kNu = {1.0, 1.1, 1.2, 1.3};  // only ratios matter
constexpr std::array<float, kScales> kScaleSigmas = {0.0f, 1.5f, 3.0f, 6.0f};
// The scale kernels are sum-normalized, so blurring attenuates a source of
// combined width sigma_c by sigma_c^2 / (sigma_c^2 + sigma_s^2). The biases
// must overcome that attenuation for each extended source on its matched
// scale, while staying low enough that point sources and smaller sources keep
// preferring their own scale. Solving those inequalities for this scene gives
// a window per scale; the values below sit mid-window (~7-18% margin on every
// competing pair), avoiding near-tie selections that would flap across GPUs.
constexpr std::array<float, kScales> kScaleBias = {1.0f, 1.45f, 2.3f, 4.2f};

struct source {
  int row, col;
  double flux;   // integrated flux at the reference frequency
  double sigma;  // 0 = point source
  double alpha;  // spectral index: flux(nu) = flux * (nu/nu0)^alpha
};

// >= 32 px from every edge, >= 40 px apart.
//
// Expected cleaning: the clean loop lands on scales 0, 2 and 3; scale 1 ends
// with ZERO components and the baseline pins that. Its static bias window is
// razor thin (a mildly extended source that scale 1 likes, scale 2 likes
// almost as much at higher bias), and the inner loop thresholds globally over
// the selected plane, so neighboring scales' sweeps absorb scale 1's niche
// before it ever holds the global argmax. The s>0 clean-loop machinery is the
// same code for every scale index, so scales 2/3 cover it.
constexpr std::array<source, 5> kSources = {{
    {64, 64, 10.0, 0.0, -0.7},
    {64, 192, 4.0, 0.0, 0.3},
    {128, 128, 6.0, 1.5, -0.5},
    {192, 64, 8.0, 3.0, 0.0},
    {192, 192, 5.0, 6.0, -1.0},
}};

// Unit-peak Gaussian PSF, identical across channels, centered at (H/2, W/2).
std::vector<float> make_psfs()
{
  std::vector<float> psf(static_cast<std::size_t>(kFreq) * kNpix);
  for (int r = 0; r < kNrow; ++r) {
    for (int c = 0; c < kNcol; ++c) {
      const double d2 =
          (r - kNrow / 2) * static_cast<double>(r - kNrow / 2) + (c - kNcol / 2) * static_cast<double>(c - kNcol / 2);
      const float v = static_cast<float>(std::exp(-d2 / (2.0 * kSigmaPsf * kSigmaPsf)));
      for (int f = 0; f < kFreq; ++f) psf.at(static_cast<std::size_t>(f) * kNpix + fdtest::flat(r, c, kNcol)) = v;
    }
  }
  return psf;
}

// Analytic dirty image: sky (sum-normalized Gaussians / deltas) convolved with
// the unit-peak Gaussian PSF has the closed form
//   dirty_src = flux * (nu/nu0)^alpha * [sigma_psf^2 / (sigma_src^2 + sigma_psf^2)]
//               * unit-peak-Gaussian(sqrt(sigma_src^2 + sigma_psf^2))
// so no convolution code (host or device) is involved in building the input.
std::vector<float> make_dirty(std::mt19937& rng)
{
  std::vector<float> dirty(static_cast<std::size_t>(kFreq) * kNpix, 0.0f);
  for (int f = 0; f < kFreq; ++f) {
    for (const auto& s : kSources) {
      const double spec = std::pow(kNu.at(f) / kNu.at(0), s.alpha);
      const double var_comb = s.sigma * s.sigma + kSigmaPsf * kSigmaPsf;
      const double amplitude = s.flux * (kSigmaPsf * kSigmaPsf / var_comb) * spec;
      const int radius = static_cast<int>(std::ceil(6.0 * std::sqrt(var_comb)));
      for (int r = std::max(0, s.row - radius); r <= std::min(kNrow - 1, s.row + radius); ++r) {
        for (int c = std::max(0, s.col - radius); c <= std::min(kNcol - 1, s.col + radius); ++c) {
          const double d2 = (r - s.row) * static_cast<double>(r - s.row) + (c - s.col) * static_cast<double>(c - s.col);
          dirty.at(static_cast<std::size_t>(f) * kNpix + fdtest::flat(r, c, kNcol)) +=
              static_cast<float>(amplitude * std::exp(-d2 / (2.0 * var_comb)));
        }
      }
    }
  }
  fdtest::add_normal_noise(rng, dirty, kNoiseSigma);
  return dirty;
}

// ---- Metric comparison machinery

struct metric_row {
  std::string name;
  double baseline;
  double actual;
  double tol_rel;
  double tol_abs;

  bool pass() const { return std::abs(actual - baseline) <= std::max(tol_abs, tol_rel * std::abs(baseline)); }
};

}  // namespace

class DdmscNonReg : public fdtest::BackendTest {};

TEST_F(DdmscNonReg, SyntheticSceneMatchesBaselineMetrics)
{
  // ---- Build the scene on the host.
  std::mt19937 rng(kSeed);
  auto h_dirty = make_dirty(rng);
  auto h_psfs = make_psfs();

  std::vector<float> h_xdes(kFreq * kOrder);
  for (int f = 0; f < kFreq; ++f) {
    h_xdes.at(f * kOrder + 0) = 1.0f;
    h_xdes.at(f * kOrder + 1) = static_cast<float>(std::log(kNu.at(f) / kNu.at(0)));
  }

  // ---- Upload the per-run inputs; the context copies its static inputs itself.
  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_dirty(sr, h_dirty);
  fdtest::device_buffer<float> d_jones(sr, std::vector<float>(static_cast<std::size_t>(kFreq) * kNpix, 1.0f));
  fdtest::device_buffer<float> d_weights(sr, std::vector<float>(kFreq, 1.0f / kFreq));

  auto h_mask = std::make_unique<bool[]>(kNpix);
  std::vector<float> h_sigmas(kScaleSigmas.begin(), kScaleSigmas.end());
  std::vector<float> scale_bias(kScaleBias.begin(), kScaleBias.end());
  std::vector<int> map_pixel_facet(kNpix, 0);

  // Static inputs stay on the host; the context stages them to the device.
  core::host_span4d<float> psfs_view(h_psfs.data(), kFacets, kFreq, kNrow, kNcol);
  core::host_span2d<float> xdes_view(h_xdes.data(), kFreq, kOrder);
  core::host_span2d<bool> mask_view(h_mask.get(), kNrow, kNcol);
  core::host_span1d<float> sigmas_view(h_sigmas.data(), kScales);
  core::host_span1d<float> bias_view(scale_bias.data(), kScales);
  core::host_span2d<int> map_view(map_pixel_facet.data(), kNrow, kNcol);

  ddmsc::context ctx(/*exec_device=*/0, psfs_view, xdes_view, mask_view, sigmas_view, bias_view, map_view, kNrow, kNcol,
                     kFreq, /*fft_padding=*/1.5f);

  const ddmsc::params p{
      // Gaussian-component cleaning shrinks the residual by only ~5% of the
      // local peak per iteration on the largest scale, so give the run ample
      // headroom: it must stop on flux, never on this budget.
      .max_iteration = 2000,
      .divergence_factor = 4.0f,
      .flux_threshold = 0.06f,  // ~3x the expected noise peak: the run must stop on flux, not budget
      .stop_rms_factor = 0.0f,
      .stop_peak_factor = 0.0f,
      .stop_cycle_factor = 0.0f,
      .stop_sidelobe_level = 0.0f,
      .clean_negative = false,
      .peak_factor = 0.15f,
      .gamma = 0.1f,
      .max_clean_iteration = 50,
      // Realistic stall tracking is load-bearing here: the large scale-3 bias
      // makes the smoothed plane outbid compact scales even when scale-3
      // cleaning stops improving the rms, and retirement is what breaks that
      // spiral (as in production). Productive outer iterations move the rms by
      // ~1e-3 in this scene; below 1e-4 counts as a stall strike.
      .scale_stall_threshold = 1e-4f,
      .enable_auto_mask = false,
      .force_enable_auto_mask = false,
      .auto_mask_peak_threshold = std::nullopt,
      .auto_mask_rms_threshold = std::nullopt,
  };

  core::span3d<float> dirty_view(d_dirty.get(), kFreq, kNrow, kNcol);
  core::span3d<float> jones_view(d_jones.get(), kFreq, kNrow, kNcol);
  core::span1d<float> weights_view(d_weights.get(), kFreq);

  // ---- Run the full minor-cycle driver. `dirty` is left as the residual.
  const auto result = ddmsc::run_ddmsc_cycles(ctx, p, dirty_view, jones_view, weights_view);

  h_dirty = d_dirty.to_host();

  // ---- Derive scalar metrics.
  const auto residual = fdtest::weighted_sum(h_dirty, std::vector<float>(kFreq, 1.0f / kFreq), kNpix);
  const float residual_rms = fdtest::std_all(residual);
  float residual_max_abs = 0.0f;
  for (const float v : residual) residual_max_abs = std::max(residual_max_abs, std::abs(v));

  const int n_components = static_cast<int>(result.peak_coords.size());
  ASSERT_EQ(result.gains.size(), result.peak_coords.size());
  ASSERT_EQ(result.coeffs.size(), result.peak_coords.size());

  // Peak-flux removed from the dirty image per component: the subtraction is
  // coeff * gain * conv_psf where conv_psf is sum-normalized with peak p_s and
  // gain = gamma / p_s by construction — so the removed peak flux is exactly
  // gamma * coeff0 on EVERY scale. (gain * coeff0 would over-count extended
  // scales by 1/p_s.)
  double cleaned_flux_total = 0.0;
  std::array<int, kScales> components_per_scale{};
  for (int i = 0; i < n_components; ++i) {
    cleaned_flux_total += static_cast<double>(p.gamma) * result.coeffs.at(i).at(0);
    components_per_scale.at(result.scales.at(i))++;
  }

  // Per-source recovered flux (sum of gamma * order-0 coeff within 15 px) and
  // distance from each planted source to its nearest component.
  std::array<double, kSources.size()> recovered_flux{};
  std::array<double, kSources.size()> nearest_component_px{};
  for (std::size_t s = 0; s < kSources.size(); ++s) {
    double nearest = 1e9;
    for (int i = 0; i < n_components; ++i) {
      const double dr = result.peak_coords.at(i).first - kSources.at(s).row;
      const double dc = result.peak_coords.at(i).second - kSources.at(s).col;
      const double dist = std::sqrt(dr * dr + dc * dc);
      nearest = std::min(nearest, dist);
      if (dist <= 15.0) recovered_flux.at(s) += static_cast<double>(p.gamma) * result.coeffs.at(i).at(0);
    }
    nearest_component_px.at(s) = nearest;
  }

  // ---- Ground-truth checks (independent of the baseline).
  ASSERT_GT(n_components, 0) << "the run cleaned nothing";
  for (std::size_t s = 0; s < kSources.size(); ++s)
    EXPECT_LE(nearest_component_px.at(s), 3.0) << "no component near planted source " << s;
  // Point sources are exact in the peak-flux bookkeeping: recovered ≈ planted.
  EXPECT_NEAR(recovered_flux.at(0), kSources.at(0).flux, 0.15 * kSources.at(0).flux);
  EXPECT_NEAR(recovered_flux.at(1), kSources.at(1).flux, 0.15 * kSources.at(1).flux);

  // ---- Assemble the metrics JSON.
  json config = {{"seed", kSeed},
                 {"nrow", kNrow},
                 {"ncol", kNcol},
                 {"n_freq", kFreq},
                 {"n_scales", kScales},
                 {"sigma_psf", kSigmaPsf},
                 {"noise_sigma", kNoiseSigma},
                 {"scale_bias", kScaleBias},
                 {"scale_stall_threshold", p.scale_stall_threshold}};
  json metrics = {
      {"n_components", n_components},
      {"total_iterations", result.total_iterations},
      {"status", std::string(common::to_string(result.status))},
      {"final_flux", result.final_flux},
      {"stop_flux", result.stop_flux},
      {"residual_rms", residual_rms},
      {"residual_max_abs", residual_max_abs},
      {"cleaned_flux_total", cleaned_flux_total},
      {"components_per_scale", components_per_scale},
      {"recovered_flux_per_source", recovered_flux},
  };

  const std::string baseline_path = FAST_DECONV_BASELINE_PATH;

  // ---- Regeneration mode: write and skip.
  const char* update = std::getenv("FAST_DECONV_UPDATE_BASELINE");
  if (update != nullptr && std::string(update) != "0") {
    json out = {{"schema_version", 1},
                {"note", "regenerate with FAST_DECONV_UPDATE_BASELINE=1; tolerances live in test_ddmsc_nonreg.cpp"},
                {"config", config},
                {"metrics", metrics}};
    std::ofstream f(baseline_path);
    ASSERT_TRUE(f.good()) << "cannot write baseline at " << baseline_path;
    f << out.dump(2) << "\n";
    GTEST_SKIP() << "baseline regenerated at " << baseline_path << " — review and commit the diff";
  }

  // ---- Load and validate the baseline.
  std::ifstream f(baseline_path);
  if (!f.good())
    FAIL() << "baseline not found at " << baseline_path
           << "\nGenerate it with: FAST_DECONV_UPDATE_BASELINE=1 ctest -L NONREG, then commit it.";
  json baseline = json::parse(f);

  ASSERT_EQ(baseline.at("config"), config) << "scene config changed since the baseline was generated — regenerate it "
                                              "(FAST_DECONV_UPDATE_BASELINE=1) and commit the diff";

  const json& base = baseline.at("metrics");

  // Categorical, so it is compared exactly instead of through the tolerance table.
  EXPECT_EQ(base.at("status").get<std::string>(), std::string(common::to_string(result.status)))
      << "the run ended for a different reason than the baseline";

  // Tolerances are code, not baseline content: regeneration can never clobber
  // them. Loose on purpose — cross-GPU / cuFFT-version drift reorders near-tie
  // argmax and scale-selection decisions.
  std::vector<metric_row> rows = {
      {"n_components", base.at("n_components").get<double>(), static_cast<double>(n_components), 0.15, 0.0},
      {"total_iterations", base.at("total_iterations").get<double>(), static_cast<double>(result.total_iterations),
       0.15, 0.0},
      {"final_flux", base.at("final_flux").get<double>(), result.final_flux, 0.10, 1e-4},
      {"stop_flux", base.at("stop_flux").get<double>(), result.stop_flux, 1e-6, 0.0},
      {"residual_rms", base.at("residual_rms").get<double>(), residual_rms, 0.10, 0.0},
      {"residual_max_abs", base.at("residual_max_abs").get<double>(), residual_max_abs, 0.10, 0.0},
      {"cleaned_flux_total", base.at("cleaned_flux_total").get<double>(), cleaned_flux_total, 0.05, 0.0},
  };
  for (int s = 0; s < kScales; ++s)
    rows.push_back({"components_per_scale[" + std::to_string(s) + "]",
                    base.at("components_per_scale").at(s).get<double>(),
                    static_cast<double>(components_per_scale.at(s)), 0.25, 3.0});
  for (std::size_t s = 0; s < kSources.size(); ++s)
    rows.push_back({"recovered_flux_per_source[" + std::to_string(s) + "]",
                    base.at("recovered_flux_per_source").at(s).get<double>(), recovered_flux.at(s), 0.10, 0.05});

  for (const auto& r : rows) EXPECT_TRUE(r.pass()) << r.name << ": baseline=" << r.baseline << " actual=" << r.actual;
}
