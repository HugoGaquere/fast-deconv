#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <fast_deconv/algorithm/ddmsc.hpp>
#include <fast_deconv/common/convergence.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <filesystem>
#include <string>

#include "ddmsc_dump.hpp"
#include "npy_loader.hpp"

// Replays a captured DDMSC session cycle by cycle on one Ddmsc and checks each cycle against kExpected.

namespace common = fast_deconv::common;
namespace core = fast_deconv::core;
namespace ddmsc = fast_deconv::algorithm::ddmsc;

namespace {

constexpr int kScales = 4;

struct cycle_result {
  common::convergence_status status;
  int outer_iterations;
  std::array<int, kScales> components_per_scale;
  float final_flux;
  float final_rms;
  double coeff0_sum;  // sum of the order-0 spectral coefficients of the cycle's components
};

// Regenerate after an intentional change: run the test and paste the rows it prints.
constexpr std::array<cycle_result, 5> kExpected = {{
    {common::convergence_status::converged, 10, {2, 80, 2, 0}, 2.748056f, 0.08433038f, 482.72864},
    {common::convergence_status::converged, 67, {57, 110, 36, 90}, 0.9711169f, 0.07559972f, 309.931593},
    {common::convergence_status::converged, 139, {393, 748, 39, 73}, 0.6188014f, 0.07324928f, 380.704849},
    {common::convergence_status::converged, 64, {43, 415, 14, 108}, 0.5710051f, 0.07210074f, 73.9196213},
    {common::convergence_status::converged, 39, {10, 35, 12, 101}, 0.596601f, 0.07194155f, 10.7770987},
}};

// Relative tolerance on the float metrics: backends and compilers round differently.
constexpr double kRelTol = 1e-4;

}  // namespace

TEST(DdmscNonReg, LofarLbaPointSourcesReplayMatchesExpected)
{
  const std::string dir = FAST_DECONV_NONREG_DUMP_DIR;
  if (!std::filesystem::exists(dir + "/init")) GTEST_SKIP() << "no dump in " << dir;

  // Held for the whole test: the session keeps host views on them.
  auto load_init = [&](const char* name) { return npy::load_npy(dir + "/init/" + name + ".npy"); };
  auto raw_psfs = load_init("raw_psfs");
  auto xdes = load_init("xdes");
  auto mask = load_init("mask");
  auto scale_sigmas = load_init("scale_sigmas");
  auto scale_bias = load_init("scale_bias");
  auto map_pixel_facet = load_init("map_pixel_facet");

  const auto n_freq = raw_psfs.shape.at(1);
  const auto nrow = map_pixel_facet.shape.at(0);
  const auto ncol = map_pixel_facet.shape.at(1);
  ASSERT_EQ(scale_sigmas.shape.at(0), static_cast<std::size_t>(kScales));

  ddmsc::Ddmsc session(
      core::host_span4d<float>(raw_psfs.as_float32(), raw_psfs.shape.at(0), n_freq, raw_psfs.shape.at(2),
                               raw_psfs.shape.at(3)),
      core::host_span2d<float>(xdes.as_float32(), n_freq, xdes.shape.at(1)),
      core::host_span2d<bool>(mask.as_bool(), nrow, ncol), core::host_span1d<float>(scale_sigmas.as_float32(), kScales),
      core::host_span1d<float>(scale_bias.as_float32(), kScales),
      core::host_span2d<int>(map_pixel_facet.as_int32(), nrow, ncol), static_cast<int>(nrow), static_cast<int>(ncol));

  for (std::size_t c = 0; c < kExpected.size(); ++c) {
    SCOPED_TRACE("cycle " + std::to_string(c));
    const std::string cdir = dir + "/cycle_" + std::to_string(c) + "/";
    auto dirty = npy::load_npy(cdir + "dirty.npy");
    auto jones_norm = npy::load_npy(cdir + "jones_norm.npy");
    auto weights = npy::load_npy(cdir + "weights_freq.npy");

    session.cfg = ddmsc_dump::read_params(cdir);
    const auto r = session.run(core::host_span3d<float>(dirty.as_float32(), n_freq, nrow, ncol),
                               core::host_span3d<const float>(jones_norm.as_float32(), n_freq, nrow, ncol),
                               core::host_span1d<const float>(weights.as_float32(), n_freq));

    cycle_result actual{r.status, r.outer_iterations, {}, r.final_flux, r.final_rms, 0.0};
    for (const int s : r.scales) actual.components_per_scale.at(s)++;
    for (std::size_t i = 0; i < r.n_components(); ++i) actual.coeff0_sum += r.coeffs.at(i * r.n_order);

    const auto& p = actual.components_per_scale;
    std::printf("    {common::convergence_status::%s, %d, {%d, %d, %d, %d}, %.7gf, %.7gf, %.9g},\n",
                std::string(common::to_string(actual.status)).c_str(), actual.outer_iterations, p[0], p[1], p[2], p[3],
                actual.final_flux, actual.final_rms, actual.coeff0_sum);

    const auto& e = kExpected.at(c);
    EXPECT_EQ(actual.status, e.status);
    EXPECT_EQ(actual.outer_iterations, e.outer_iterations);
    EXPECT_EQ(actual.components_per_scale, e.components_per_scale);
    EXPECT_NEAR(actual.final_flux, e.final_flux, kRelTol * e.final_flux);
    EXPECT_NEAR(actual.final_rms, e.final_rms, kRelTol * e.final_rms);
    EXPECT_NEAR(actual.coeff0_sum, e.coeff0_sum, kRelTol * std::abs(e.coeff0_sum));
  }
}
