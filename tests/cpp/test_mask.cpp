#include <gtest/gtest.h>

#include <algorithm>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <memory>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/host_oracles.hpp"

namespace core = fast_deconv::core;
namespace common = fast_deconv::common;
namespace fdtest = fast_deconv::test;

using fdtest::flat;

// build_auto_mask output: (n_scales, nrow, ncol), 1 = masked, 0 = valid.
class BuildAutoMask : public fdtest::BackendTest {
 protected:
  // Single-channel run with a square PSF. The output starts all masked, as left by an earlier cycle, so a
  // valid pixel the function forgets to write shows up.
  std::vector<int> run(const std::vector<common::index2d>& coords, const std::vector<int>& scales,
                       const std::vector<float>& psf, int psf_size, const std::vector<float>& sigmas,
                       const bool* external, int nrow, int ncol)
  {
    const int n_scales = static_cast<int>(sigmas.size()), n = n_scales * nrow * ncol;
    const auto mask = std::make_unique<bool[]>(n);
    std::fill_n(mask.get(), n, true);
    const std::vector<float> weights = {1.0f};

    const auto sr = res().make_ctx();
    const auto d_psf = sr.stage(core::host_span3d<const float>(psf.data(), 1, psf_size, psf_size));
    const auto d_weights = sr.stage(core::host_span1d<const float>(weights.data(), 1));
    const auto d_external = sr.stage(core::host_span2d<const bool>(external, nrow, ncol));
    auto d_mask = sr.stage(core::host_span3d<bool>(mask.get(), n_scales, nrow, ncol));

    common::build_auto_mask(sr, coords, scales, d_psf, d_weights, sigmas, d_external, d_mask);
    sr.unstage(d_mask, mask.get());
    sr.wait();
    return {mask.get(), mask.get() + n};
  }
};

// Delta PSF and sigma 0: the FWHM support is a single pixel, so on each scale the only valid pixels are the
// peaks of that scale's own components.
TEST_F(BuildAutoMask, DeltaPsfLeavesOnlyEachScalesPeaksValid)
{
  const int nrow = 4, ncol = 5, plane = nrow * ncol;
  std::vector<float> psf(8 * 8, 0.0f);
  psf.at(flat(4, 4, 8)) = 1.0f;  // the FFT center of an 8x8 plane
  const auto no_external = std::make_unique<bool[]>(plane);

  const auto mask = run({{0, 0}, {1, 2}, {3, 4}}, /*scales=*/{0, 2, 1}, psf, 8, /*sigmas=*/{0.0f, 0.0f, 0.0f},
                        no_external.get(), nrow, ncol);

  std::vector<int> expected(3 * plane, 1);
  expected.at(0 * plane + flat(0, 0, ncol)) = 0;
  expected.at(2 * plane + flat(1, 2, ncol)) = 0;
  expected.at(1 * plane + flat(3, 4, ncol)) = 0;
  EXPECT_EQ(mask, expected);
}

// Gaussian PSF (sigma 1.5), component on scale 1 (sigma 1): conv2 = PSF ** G_1 ** G_1 is a Gaussian of
// sigma sqrt(1.5^2 + 2 * 1^2) ~ 2.06, which is above half its peak within ~2.4 px. That neighborhood is
// valid, except where the external mask covers it; everything else, and the component-free scale 0, is masked.
TEST_F(BuildAutoMask, GaussianPsfValidatesTheFwhmNeighborhood)
{
  const int n = 16, plane = n * n;
  const auto external = std::make_unique<bool[]>(plane);
  external[flat(8, 9, n)] = true;

  const auto mask = run({{8, 8}}, /*scales=*/{1}, fdtest::gaussian2d(n, n, 8, 8, 1.5), n, /*sigmas=*/{0.0f, 1.0f},
                        external.get(), n, n);
  const auto scale1 = [&](int r, int c) { return mask.at(plane + flat(r, c, n)); };

  EXPECT_EQ(scale1(8, 8), 0);  // the component
  EXPECT_EQ(scale1(7, 8), 0);  // 1 px away
  EXPECT_EQ(scale1(8, 6), 0);  // 2 px away
  EXPECT_EQ(scale1(8, 5), 1);  // 3 px away: outside the FWHM
  EXPECT_EQ(scale1(0, 0), 1);
  EXPECT_EQ(scale1(8, 9), 1);  // inside the FWHM but externally masked

  EXPECT_EQ(std::count(mask.begin(), mask.begin() + plane, 1), plane) << "scale 0 has no component";
}

// combine_scale_masks output: (nrow, ncol), 1 = masked, 0 = valid.
class CombineScaleMasks : public fdtest::BackendTest {
 protected:
  // The output starts all valid, so a masked pixel the function forgets to write shows up.
  std::vector<int> run(const std::vector<bool>& masks, int n_scales, int nrow, int ncol, const std::vector<int>& scales)
  {
    const int n = n_scales * nrow * ncol, plane = nrow * ncol;
    const auto h_masks = std::make_unique<bool[]>(n);
    std::copy(masks.begin(), masks.end(), h_masks.get());
    const auto out = std::make_unique<bool[]>(plane);

    const auto sr = res().make_ctx();
    const auto d_masks = sr.stage(core::host_span3d<const bool>(h_masks.get(), n_scales, nrow, ncol));
    auto d_out = sr.stage(core::host_span2d<bool>(out.get(), nrow, ncol));

    common::combine_scale_masks(sr, d_masks, scales, d_out);
    sr.unstage(d_out, out.get());
    sr.wait();
    return {out.get(), out.get() + plane};
  }
};

// Three 1x4 scale masks, each valid on a different pixel: a pixel is valid when one of the chosen scales is.
TEST_F(CombineScaleMasks, ValidWhereAnyChosenScaleIsValid)
{
  // clang-format off
  const std::vector<bool> masks = {
      false, true,  true,  true,   // scale 0: pixel 0 valid
      true,  false, true,  true,   // scale 1: pixel 1 valid
      true,  true,  false, true,   // scale 2: pixel 2 valid
  };
  // clang-format on
  EXPECT_EQ(run(masks, 3, 1, 4, {0, 1, 2}), (std::vector<int>{0, 0, 0, 1}));
  EXPECT_EQ(run(masks, 3, 1, 4, {0, 2}), (std::vector<int>{0, 1, 0, 1})) << "scale 1 left out";
  EXPECT_EQ(run(masks, 3, 1, 4, {1}), (std::vector<int>{1, 0, 1, 1}));
}

TEST_F(CombineScaleMasks, NoScalesMasksEverything)
{
  const std::vector<bool> masks(2 * 3, false);
  EXPECT_EQ(run(masks, 2, 1, 3, {}), (std::vector<int>{1, 1, 1}));
}
