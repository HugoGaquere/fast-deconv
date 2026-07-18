#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <fast_deconv/algorithm/wscms_class.hpp>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <memory>
#include <optional>
#include <vector>

#include "helpers/device_buffers.hpp"
#include "helpers/gpu_test.hpp"

namespace core = fast_deconv::core;
namespace wscms = fast_deconv::algorithm::wscms;
namespace fdtest = fast_deconv::test;

// Minimal but real construction scene: the Wscms constructor allocates the
// pool, both streams, and the cuFFT plans, so this is a GPU fixture. All spans
// are stored as views by the class — the backing buffers live in the fixture.
class WscmsClass : public fdtest::GpuTest {
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

    sr_ = std::make_unique<holder>(res());

    std::vector<float> psfs(kFacets * kFreq * kSize * kSize, 0.0f);
    for (int f = 0; f < kFreq; ++f) psfs.at(f * kSize * kSize + (kSize / 2) * kSize + kSize / 2) = 1.0f;
    d_psfs_ = std::make_unique<fdtest::device_buffer<float>>(res(), sr_->sr, psfs);

    d_xdes_ =
        std::make_unique<fdtest::device_buffer<float>>(res(), sr_->sr, std::vector<float>{1.0f, 0.0f, 1.0f, 0.1f});
    d_mask_ = std::make_unique<fdtest::device_buffer<bool>>(res(), sr_->sr, std::vector<bool>(kSize * kSize, false));
    d_sigmas_ = std::make_unique<fdtest::device_buffer<float>>(res(), sr_->sr, std::vector<float>{0.0f, 1.5f});
    scale_bias_ = {1.0f, 0.8f};
    map_pixel_facet_ = std::vector<int>(kSize * kSize, 0);
  }

  wscms::Wscms make_wscms()
  {
    core::device_span4d<float> psf_view(d_psfs_->get(), kFacets, kFreq, kSize, kSize);
    core::device_span2d<float> xdes_view(d_xdes_->get(), kFreq, kOrder);
    core::device_span2d<bool> mask_view(d_mask_->get(), kSize, kSize);
    core::device_vect<float> sigma_view(d_sigmas_->get(), kScales);
    core::host_vect<float> bias_view(scale_bias_.data(), kScales);
    core::host_span2d<int> map_view(map_pixel_facet_.data(), kSize, kSize);

    return wscms::Wscms(psf_view, xdes_view, mask_view, sigma_view, bias_view, map_view,
                        /*dirty_nrow=*/kSize, /*dirty_ncol=*/kSize, /*n_freq=*/kFreq, /*fft_padding=*/1.5f);
  }

 private:
  // stream_resources is non-movable; hold it behind a pointer so SetUp can
  // create it after the GPU check.
  struct holder {
    explicit holder(core::resources& r) : sr(r.make_stream()) {}
    core::stream_resources sr;
  };
  std::unique_ptr<holder> sr_;
  std::unique_ptr<fdtest::device_buffer<float>> d_psfs_;
  std::unique_ptr<fdtest::device_buffer<float>> d_xdes_;
  std::unique_ptr<fdtest::device_buffer<bool>> d_mask_;
  std::unique_ptr<fdtest::device_buffer<float>> d_sigmas_;
  std::vector<float> scale_bias_;
  std::vector<int> map_pixel_facet_;
};

// The constructor builds the pool, both streams, the shared cuFFT work area and
// all plans — surviving construction and destruction on a tiny scene is the
// smoke test.
TEST_F(WscmsClass, ConstructsAndDestroysCleanly)
{
  auto w = make_wscms();
  (void)w;
}

TEST_F(WscmsClass, SetterGetterRoundTripForEveryParameter)
{
  auto w = make_wscms();

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

// wscms_result::add_coeffs_from_device slices a (n_components, n_order) device
// buffer into one host vector per component.
TEST_F(WscmsClass, AddCoeffsFromDeviceSlicesRows)
{
  const int n_components = 3, n_order = 2;
  const std::vector<float> coeffs = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  const auto sr = res().make_stream();
  fdtest::device_buffer<float> d_coeffs(res(), sr, coeffs);
  core::device_span2d<float> view(d_coeffs.get(), n_components, n_order);

  wscms::wscms_result result(/*max_iter=*/10, n_order);
  result.add_coeffs_from_device(view);

  ASSERT_EQ(result.coeffs.size(), static_cast<std::size_t>(n_components));
  for (int i = 0; i < n_components; ++i) {
    ASSERT_EQ(result.coeffs.at(i).size(), static_cast<std::size_t>(n_order));
    EXPECT_FLOAT_EQ(result.coeffs.at(i).at(0), coeffs.at(i * n_order + 0));
    EXPECT_FLOAT_EQ(result.coeffs.at(i).at(1), coeffs.at(i * n_order + 1));
  }
}
