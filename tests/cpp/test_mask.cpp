#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/device_buffers.hpp"
#include "helpers/host_oracles.hpp"

namespace core = fast_deconv::core;
namespace common = fast_deconv::common;
namespace fdtest = fast_deconv::test;

using fdtest::flat;

// build_auto_mask output: (n_scales, nrow, ncol), 1 = masked, 0 = valid.
class BuildAutoMask : public fdtest::BackendTest {
 protected:
  // Single-channel run with a square PSF. The output starts as garbage (0xff), so a pixel the function
  // forgets to write cannot pass for either 0 or 1.
  std::vector<uint8_t> run(const std::vector<common::index2d>& coords, const std::vector<int>& scales,
                           const std::vector<float>& psf, int psf_size, const std::vector<float>& sigmas,
                           const std::vector<bool>& external, int nrow, int ncol)
  {
    const auto sr = res().make_ctx();
    const int n_scales = static_cast<int>(sigmas.size());
    fdtest::device_buffer<bool> d_mask(sr, static_cast<std::size_t>(n_scales) * nrow * ncol);
    d_mask.fill_bytes(0xff);
    fdtest::device_buffer<float> d_psf(sr, psf), d_weights(sr, std::vector<float>{1.0f});
    fdtest::device_buffer<bool> d_external(sr, external);

    common::build_auto_mask(sr, coords, scales, core::span3d<const float>(d_psf.get(), 1, psf_size, psf_size),
                            core::span1d<const float>(d_weights.get(), 1), sigmas,
                            core::span2d<const bool>(d_external.get(), nrow, ncol),
                            core::span3d<bool>(d_mask.get(), n_scales, nrow, ncol));
    return d_mask.to_host();
  }
};

// Delta PSF and sigma 0: the FWHM support is a single pixel, so on each scale the only valid pixels are the
// peaks of that scale's own components.
TEST_F(BuildAutoMask, DeltaPsfLeavesOnlyEachScalesPeaksValid)
{
  const int nrow = 4, ncol = 5, plane = nrow * ncol;
  std::vector<float> psf(8 * 8, 0.0f);
  psf.at(flat(4, 4, 8)) = 1.0f;  // the FFT center of an 8x8 plane

  const auto mask = run({{0, 0}, {1, 2}, {3, 4}}, /*scales=*/{0, 2, 1}, psf, 8, /*sigmas=*/{0.0f, 0.0f, 0.0f},
                        std::vector<bool>(plane, false), nrow, ncol);

  std::vector<uint8_t> expected(3 * plane, 1);
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
  std::vector<bool> external(plane, false);
  external.at(flat(8, 9, n)) = true;

  const auto mask =
      run({{8, 8}}, /*scales=*/{1}, fdtest::gaussian2d(n, n, 8, 8, 1.5), n, /*sigmas=*/{0.0f, 1.0f}, external, n, n);
  const auto scale1 = [&](int r, int c) { return mask.at(plane + flat(r, c, n)); };

  EXPECT_EQ(scale1(8, 8), 0);  // the component
  EXPECT_EQ(scale1(7, 8), 0);  // 1 px away
  EXPECT_EQ(scale1(8, 6), 0);  // 2 px away
  EXPECT_EQ(scale1(8, 5), 1);  // 3 px away: outside the FWHM
  EXPECT_EQ(scale1(0, 0), 1);
  EXPECT_EQ(scale1(8, 9), 1);  // inside the FWHM but externally masked

  EXPECT_EQ(std::count(mask.begin(), mask.begin() + plane, 1), plane) << "scale 0 has no component";
}
