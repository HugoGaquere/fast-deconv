#include <gtest/gtest.h>

#include <fast_deconv/algorithm/ddmsc.hpp>
#include <fast_deconv/algorithm/ddmsc_types.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/device_buffers.hpp"

namespace core = fast_deconv::core;
namespace ddmsc = fast_deconv::algorithm::ddmsc;
namespace fdtest = fast_deconv::test;

// Minimal but real construction scene: the context builds the pool and both
// streams and stages the static inputs at construction. It keeps views of the
// host spans, so the backing buffers live in the fixture.
class DdmscClass : public fdtest::BackendTest {
 protected:
  static constexpr int kFacets = 1;
  static constexpr int kFreq = 2;
  static constexpr int kSize = 32;  // dirty and PSF are both 32x32
  static constexpr int kScales = 2;
  static constexpr int kOrder = 2;

  void SetUp() override
  {
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
    core::host_span1d<float> sigma_view(sigmas_.data(), kScales);
    core::host_span1d<float> bias_view(scale_bias_.data(), kScales);
    core::host_span2d<int> map_view(map_pixel_facet_.data(), kSize, kSize);

    return ddmsc::Ddmsc(psf_view, xdes_view, mask_view, sigma_view, bias_view, map_view,
                        /*dirty_nrow=*/kSize, /*dirty_ncol=*/kSize);
  }

 private:
  // Static inputs live on the host; the context stages them at construction.
  std::vector<float> psfs_;
  std::vector<float> xdes_;
  std::unique_ptr<bool[]> mask_;
  std::vector<float> sigmas_;
  std::vector<float> scale_bias_;
  std::vector<int> map_pixel_facet_;
};

// Construction builds the pool and both streams and stages the static inputs;
// surviving construction and destruction on a tiny scene is the smoke test.
TEST_F(DdmscClass, ConstructsAndDestroysCleanly)
{
  auto w = make_ddmsc();
  (void)w;
}

// ddmsc_result::set_coeffs copies a (n_components, n_order) device
// buffer into the flat row-major host coeffs.
TEST_F(DdmscClass, SetCoeffsCopiesRows)
{
  const int n_components = 3, n_order = 2;
  const std::vector<float> coeffs = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_coeffs(sr, coeffs);
  core::span2d<float> view(d_coeffs.get(), n_components, n_order);

  ddmsc::ddmsc_result result(/*capacity=*/10);
  for (int i = 0; i < n_components; ++i) result.add_component({i, i}, 0, 1.0f);
  const core::span2d<float> short_view(d_coeffs.get(), n_components - 1, n_order);
  EXPECT_THROW(result.set_coeffs(sr, short_view), std::logic_error);  // one row per component
  result.set_coeffs(sr, view);

  EXPECT_EQ(result.n_order, n_order);
  EXPECT_EQ(result.coeffs, coeffs);
}
