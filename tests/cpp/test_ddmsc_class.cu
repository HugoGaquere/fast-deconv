#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <fast_deconv/algorithm/ddmsc.hpp>
#include <fast_deconv/algorithm/ddmsc_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "helpers/device_buffers.hpp"
#include "helpers/gpu_test.hpp"

namespace core = fast_deconv::core;
namespace ddmsc = fast_deconv::algorithm::ddmsc;
namespace fdtest = fast_deconv::test;

// Minimal but real construction scene: the pool, both streams and the cuFFT
// plans are built on the first run(), so this is a GPU fixture. All spans are
// stored as views by the class — the backing buffers live in the fixture.
class DdmscClass : public fdtest::GpuTest {
 protected:
  static constexpr int kFacets = 1;
  static constexpr int kFreq = 2;
  static constexpr int kSize = 32;  // dirty and PSF are both 32x32
  static constexpr int kScales = 2;
  static constexpr int kOrder = 2;

  void SetUp() override
  {
    fdtest::GpuTest::SetUp();
    if (IsSkipped()) return;

    psfs_.assign(kFacets * kFreq * kSize * kSize, 0.0f);
    for (int f = 0; f < kFreq; ++f) psfs_.at(f * kSize * kSize + (kSize / 2) * kSize + kSize / 2) = 1.0f;

    xdes_ = {1.0f, 0.0f, 1.0f, 0.1f};
    mask_ = std::make_unique<bool[]>(kSize * kSize);  // value-initialized to false
    sigmas_ = {0.0f, 1.5f};
    scale_bias_ = {1.0f, 0.8f};
    map_pixel_facet_ = std::vector<int>(kSize * kSize, 0);
  }

  ddmsc::Ddmsc make_ddmsc()
  {
    core::host_span4d<float> psf_view(psfs_.data(), kFacets, kFreq, kSize, kSize);
    core::host_span2d<float> xdes_view(xdes_.data(), kFreq, kOrder);
    core::host_span2d<bool> mask_view(mask_.get(), kSize, kSize);
    core::host_vect<float> sigma_view(sigmas_.data(), kScales);
    core::host_vect<float> bias_view(scale_bias_.data(), kScales);
    core::host_span2d<int> map_view(map_pixel_facet_.data(), kSize, kSize);

    return ddmsc::Ddmsc(psf_view, xdes_view, mask_view, sigma_view, bias_view, map_view,
                        /*dirty_nrow=*/kSize, /*dirty_ncol=*/kSize, /*n_freq=*/kFreq, /*fft_padding=*/1.5f);
  }

 private:
  // Static inputs live on the host; the first run() stages them to device.
  std::vector<float> psfs_;
  std::vector<float> xdes_;
  std::unique_ptr<bool[]> mask_;
  std::vector<float> sigmas_;
  std::vector<float> scale_bias_;
  std::vector<int> map_pixel_facet_;
};

// Construction is allocation-free now; the pool, both streams, the shared cuFFT
// work area and all plans are built by the first run(). Surviving construction
// and destruction on a tiny scene is still the smoke test.
TEST_F(DdmscClass, ConstructsAndDestroysCleanly)
{
  auto w = make_ddmsc();
  (void)w;
}

TEST_F(DdmscClass, SetterGetterRoundTripForEveryParameter)
{
  auto w = make_ddmsc();

  w.set_clean_negative(true);
  EXPECT_TRUE(w.clean_negative());
  w.set_clean_negative(false);
  EXPECT_FALSE(w.clean_negative());

  w.set_peak_factor(0.35f);
  EXPECT_FLOAT_EQ(w.peak_factor(), 0.35f);

  w.set_gamma(0.07f);
  EXPECT_FLOAT_EQ(w.gamma(), 0.07f);

  w.set_max_sub_iteration(123);
  EXPECT_EQ(w.max_sub_iteration(), 123);

  w.set_flux_threshold(0.002f);
  EXPECT_FLOAT_EQ(w.flux_threshold(), 0.002f);

  w.set_stop_rms_factor(2.5f);
  EXPECT_FLOAT_EQ(w.stop_rms_factor(), 2.5f);

  w.set_stop_peak_factor(0.01f);
  EXPECT_FLOAT_EQ(w.stop_peak_factor(), 0.01f);

  w.set_stop_cycle_factor(0.75f);
  EXPECT_FLOAT_EQ(w.stop_cycle_factor(), 0.75f);

  w.set_stop_sidelobe_level(0.2f);
  EXPECT_FLOAT_EQ(w.stop_sidelobe_level(), 0.2f);

  w.set_max_iteration(4567);
  EXPECT_EQ(w.max_iteration(), 4567);

  w.set_divergence_factor(3.5f);
  EXPECT_FLOAT_EQ(w.divergence_factor(), 3.5f);

  w.set_stall_threshold(1e-5f);
  EXPECT_FLOAT_EQ(w.stall_threshold(), 1e-5f);

  w.set_auto_mask(true);
  EXPECT_TRUE(w.auto_mask());

  w.set_force_auto_mask(true);
  EXPECT_TRUE(w.force_auto_mask());

  w.set_auto_mask_peak_threshold(0.05f);
  ASSERT_TRUE(w.auto_mask_peak_threshold().has_value());
  EXPECT_FLOAT_EQ(*w.auto_mask_peak_threshold(), 0.05f);
  w.set_auto_mask_peak_threshold(std::nullopt);
  EXPECT_FALSE(w.auto_mask_peak_threshold().has_value());

  w.set_auto_mask_rms_threshold(4.0f);
  ASSERT_TRUE(w.auto_mask_rms_threshold().has_value());
  EXPECT_FLOAT_EQ(*w.auto_mask_rms_threshold(), 4.0f);
  w.set_auto_mask_rms_threshold(std::nullopt);
  EXPECT_FALSE(w.auto_mask_rms_threshold().has_value());
}

// ddmsc_result::add_coeffs_from_device slices a (n_components, n_order) device
// buffer into one host vector per component.
TEST_F(DdmscClass, AddCoeffsFromDeviceSlicesRows)
{
  const int n_components = 3, n_order = 2;
  const std::vector<float> coeffs = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  const auto sr = res().make_stream();
  fdtest::device_buffer<float> d_coeffs(res(), sr, coeffs);
  core::device_span2d<float> view(d_coeffs.get(), n_components, n_order);

  ddmsc::ddmsc_result result(/*max_iter=*/10, n_order);
  result.add_coeffs_from_device(sr, view);

  ASSERT_EQ(result.coeffs.size(), static_cast<std::size_t>(n_components));
  for (int i = 0; i < n_components; ++i) {
    ASSERT_EQ(result.coeffs.at(i).size(), static_cast<std::size_t>(n_order));
    EXPECT_FLOAT_EQ(result.coeffs.at(i).at(0), coeffs.at(i * n_order + 0));
    EXPECT_FLOAT_EQ(result.coeffs.at(i).at(1), coeffs.at(i * n_order + 1));
  }
}

// Plane guard: the context validates its dimensions and allocates nothing, so
// this runs without a GPU. 46341^2 = 2,147,488,281 is the first square past
// INT32_MAX; the spans are never dereferenced before the first run.
TEST(DdmscContextGuard, RejectsPlaneLargerThanInt32)
{
  constexpr int kBig = 46341;
  constexpr int kSmall = 8;

  core::host_span4d<float> psfs(static_cast<float*>(nullptr), 1, 1, kSmall, kSmall);
  core::host_span4d<float> big_psfs(static_cast<float*>(nullptr), 1, 1, kBig, kBig);
  core::host_span2d<float> xdes(static_cast<float*>(nullptr), 1, 2);
  core::host_span2d<bool> mask(static_cast<bool*>(nullptr), kSmall, kSmall);
  core::host_vect<float> sigmas(static_cast<float*>(nullptr), 1);
  core::host_vect<float> bias(static_cast<float*>(nullptr), 1);
  core::host_span2d<int> map(static_cast<int*>(nullptr), kSmall, kSmall);

  EXPECT_THROW(ddmsc::context(0, psfs, xdes, mask, sigmas, bias, map, kBig, kBig, 1, 1.5f), std::invalid_argument);
  EXPECT_THROW(ddmsc::context(0, big_psfs, xdes, mask, sigmas, bias, map, kSmall, kSmall, 1, 1.5f),
               std::invalid_argument);
  EXPECT_NO_THROW(ddmsc::context(0, psfs, xdes, mask, sigmas, bias, map, kSmall, kSmall, 1, 1.5f));
}
