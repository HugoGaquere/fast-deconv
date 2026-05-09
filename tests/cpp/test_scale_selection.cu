#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fast_deconv/util/utils.hpp>
#include <filesystem>
#include <limits>
#include <string>

#include "npy_loader.hpp"

namespace fs = std::filesystem;
namespace core = fast_deconv::core;
namespace scale = fast_deconv::scale;
namespace common = fast_deconv::common;
namespace matrix = fast_deconv::matrix;
namespace util = fast_deconv::util;

class ScaleSelectionTest : public ::testing::Test {
 protected:
  static fs::path data_dir()
  {
    return fs::path(__FILE__).parent_path().parent_path() / "data" / "scale_selection_reference_data";
  }

  static npy::NpyArray load(const std::string& name)
  {
    return npy::load_npy((data_dir() / (name + ".npy")).string());
  }
};

TEST_F(ScaleSelectionTest, LoadInputMeanDirty)
{
  auto arr = load("input_scale_convolve_mean_dirty");

  ASSERT_EQ(arr.ndim(), 2);
  EXPECT_EQ(arr.shape[0], 525);
  EXPECT_EQ(arr.shape[1], 525);
  EXPECT_TRUE(arr.is_float32());
}

TEST_F(ScaleSelectionTest, LoadConvMeanDirtys)
{
  auto arr = load("output_conv_mean_dirtys");

  ASSERT_EQ(arr.ndim(), 3);
  EXPECT_EQ(arr.shape[0], 7);  // n_scales
  EXPECT_EQ(arr.shape[1], 525);
  EXPECT_EQ(arr.shape[2], 525);
}

TEST_F(ScaleSelectionTest, LoadScaleSelectOutputs)
{
  auto xy = load("output_scale_select_xy");
  auto scale = load("output_scale_select_scale");
  auto peak = load("output_scale_select_peak");

  // xy should be [248, 279]
  ASSERT_EQ(xy.size(), 2);

  // scale should be 0
  ASSERT_EQ(scale.size(), 1);

  // peak should be ~2.528
  ASSERT_EQ(peak.size(), 1);
}

TEST_F(ScaleSelectionTest, MakeScales)
{
  auto sigmas_npy = load("input_scale_convolve_sigmas");  // (7,)
  auto expected_npy = load("output_kernel_ft_half");      // (7, 631, 316) natural order, half-complex

  const int n_scales = expected_npy.shape[0];
  const int freq_nrow = expected_npy.shape[1];
  const int freq_ncol = expected_npy.shape[2];
  const int full_ncol = (freq_ncol - 1) * 2 + 1;  // reconstruct padded ncol from half-complex
  const int half_total = freq_nrow * freq_ncol;

  core::resources resources(0);
  const auto& stream_res = resources.get_stream_resources();

  float* d_sigmas = resources.alloc_async<float>(n_scales, stream_res);
  float* d_scales = resources.alloc_async<float>(half_total * n_scales, stream_res);

  CHECK_CUDA(cudaMemcpyAsync(d_sigmas, sigmas_npy.as_float32(), n_scales * sizeof(float),
                             cudaMemcpyHostToDevice, stream_res.cuda_stream));
  stream_res.sync();

  core::device_vect<float> sigmas_view(d_sigmas, n_scales);
  core::device_span3d<float> scales_view(d_scales, n_scales, freq_nrow, freq_ncol);

  scale::make_gaussian_kernels_async(stream_res, sigmas_view, full_ncol, scales_view);

  std::vector<float> h_scales(half_total * n_scales);
  CHECK_CUDA(cudaMemcpyAsync(h_scales.data(), d_scales, half_total * n_scales * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_res.cuda_stream));
  stream_res.sync();

  const float* expected = expected_npy.as_float32();
  for (int i = 0; i < half_total * n_scales; i++) {
    ASSERT_NEAR(h_scales[i], expected[i], 1e-6f) << "Mismatch at flat index " << i;
  }

  resources.free_async(d_sigmas, stream_res);
  resources.free_async(d_scales, stream_res);
  stream_res.sync();
}

TEST_F(ScaleSelectionTest, ScaleConvolve)
{
  auto dirty_npy = load("input_scale_convolve_mean_dirty");  // (525, 525)
  auto kernel_npy = load("output_kernel_ft_half");           // (7, 631, 316) natural order, half-complex
  auto expected_npy = load("output_conv_mean_dirtys");       // (7, 525, 525)

  const int nrow = dirty_npy.shape[0];
  const int ncol = dirty_npy.shape[1];
  const int n_scales = kernel_npy.shape[0];
  const int freq_nrow = kernel_npy.shape[1];
  const int freq_ncol = kernel_npy.shape[2];
  const int npix = nrow * ncol;
  const int kernel_total = kernel_npy.size();
  const float padding = 1.2f;

  core::resources resources(0);
  const auto& stream_res = resources.get_stream_resources();

  fast_deconv::algorithm::wscms::scale_convolve_ctx ctx(resources, nrow, ncol, /*backward_batch_size=*/1, padding);

  float* d_dirty = resources.alloc_async<float>(npix, stream_res);
  float* d_scales = resources.alloc_async<float>(kernel_total, stream_res);
  float* d_output = resources.alloc_async<float>(npix * n_scales, stream_res);

  CHECK_CUDA(cudaMemcpyAsync(d_dirty, dirty_npy.as_float32(), npix * sizeof(float),
                             cudaMemcpyHostToDevice, stream_res.cuda_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_scales, kernel_npy.as_float32(), kernel_total * sizeof(float),
                             cudaMemcpyHostToDevice, stream_res.cuda_stream));
  stream_res.sync();

  core::device_span2d<float> dirty_view(d_dirty, nrow, ncol);
  core::device_span3d<float> scales_view(d_scales, n_scales, freq_nrow, freq_ncol);
  core::device_span3d<float> output_view(d_output, n_scales, nrow, ncol);

  scale::convolve_with_scales(stream_res, ctx, dirty_view, scales_view, output_view);

  std::vector<float> h_output(npix * n_scales);
  CHECK_CUDA(cudaMemcpyAsync(h_output.data(), d_output, npix * n_scales * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_res.cuda_stream));
  stream_res.sync();

  const float* expected = expected_npy.as_float32();
  for (int s = 0; s < n_scales; s++) {
    for (int i = 0; i < npix; i++) {
      int idx = s * npix + i;
      ASSERT_NEAR(h_output[idx], expected[idx], 1e-6f)
          << "Mismatch at scale " << s << ", pixel " << i;
    }
  }

  resources.free_async(d_dirty, stream_res);
  resources.free_async(d_scales, stream_res);
  resources.free_async(d_output, stream_res);
  stream_res.sync();
}

TEST_F(ScaleSelectionTest, ScaleSelectionResult)
{
  // Load reference data
  auto scaled_dirty_npy = load("output_conv_mean_dirtys");  // (7, 525, 525)
  auto mask_npy = load("input_scale_convolve_mask");        // (525, 525)
  auto bias_npy = load("input_scale_convolve_bias");        // (7,)

  // Expected outputs
  auto expected_xy = load("output_scale_select_xy");        // [248, 279]
  auto expected_scale = load("output_scale_select_scale");  // 0
  auto expected_peak = load("output_scale_select_peak");    // 2.528

  const int n_scales = scaled_dirty_npy.shape[0];
  const int nrow = scaled_dirty_npy.shape[1];
  const int ncol = scaled_dirty_npy.shape[2];
  const int npix = nrow * ncol;

  core::resources resources(0);
  const core::stream_resources& stream_res = resources.get_stream_resources();

  // Allocate and copy to device (bias stays on host)
  float* d_scaled_dirty = resources.alloc_async<float>(n_scales * npix, stream_res);
  bool* d_mask = resources.alloc_async<bool>(npix, stream_res);

  CHECK_CUDA(cudaMemcpyAsync(d_scaled_dirty, scaled_dirty_npy.as_float32(),
                             n_scales * npix * sizeof(float), cudaMemcpyHostToDevice,
                             stream_res.cuda_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_mask, mask_npy.as_bool(), npix * sizeof(bool),
                             cudaMemcpyHostToDevice, stream_res.cuda_stream));
  stream_res.sync();

  core::device_span3d<float> scaled_dirty_view(d_scaled_dirty, n_scales, nrow, ncol);
  core::device_span2d<bool> mask_view(d_mask, nrow, ncol);
  core::host_vect<float> bias_view(bias_npy.as_float32(), n_scales);

  // The new scale_selection expects the input to already have masked-out pixels
  // set to -inf (and absolute values for clean_negative=true). This is what
  // run_wscms_cycles does via mask_and_abs_async before calling scale_selection.
  common::mask_and_abs_async(stream_res, scaled_dirty_view, mask_view,
                             -std::numeric_limits<float>::infinity(),
                             /*abs=*/true);

  const int best_scale = scale::scale_selection(stream_res, scaled_dirty_view, bias_view,
                                                /*retired_scales=*/{});

  // Recover (row, col) and peak value from the selected slice via argmax
  matrix::argmax_workspace peak_ws{stream_res, static_cast<size_t>(npix)};
  auto selected_slice = core::slice_leading(scaled_dirty_view, best_scale);
  auto [peak_value, peak_index] = matrix::argmax(peak_ws, selected_slice.data_handle());
  const auto peak_coords = util::unravel_index_2D(peak_index, ncol);

  // Compare results
  EXPECT_EQ(best_scale, expected_scale.as_int32()[0]);
  EXPECT_EQ(peak_coords.first, expected_xy.as_int32()[0]);
  EXPECT_EQ(peak_coords.second, expected_xy.as_int32()[1]);
  EXPECT_NEAR(peak_value, expected_peak.as_float32()[0], 1e-6f);

  // Cleanup
  resources.free_async(d_scaled_dirty, stream_res);
  resources.free_async(d_mask, stream_res);
  stream_res.sync();
}

// CopyScaleSlice test: the previous detail::copy_scale_slice did a
// cudaMemcpy to extract one scale slice. The new API uses core::slice_leading
// to obtain a non-owning view into the same buffer (no copy), so a direct
// equivalent test is no longer meaningful. Kept commented for reference.
//
// TEST_F(ScaleSelectionTest, CopyScaleSlice)
// {
//   auto conv_npy = load("output_conv_mean_dirtys");          // (7, 525, 525)
//   auto expected_npy = load("output_scale_select_dirty");    // (525, 525)
//   auto expected_scale = load("output_scale_select_scale");  // 0
//
//   const int n_scales = conv_npy.shape[0];
//   const int nrow = conv_npy.shape[1];
//   const int ncol = conv_npy.shape[2];
//   const int npix = nrow * ncol;
//   const int best_scale = expected_scale.as_int32()[0];
//
//   core::resources resources(0);
//   const auto& stream_res = resources.get_stream_resources();
//
//   float* d_src = resources.alloc_async<float>(n_scales * npix, stream_res);
//   CHECK_CUDA(cudaMemcpyAsync(d_src, conv_npy.as_float32(), n_scales * npix * sizeof(float),
//                              cudaMemcpyHostToDevice, stream_res.cuda_stream));
//   stream_res.sync();
//
//   core::device_span3d<float> src_view(d_src, n_scales, nrow, ncol);
//   auto slice = core::slice_leading(src_view, best_scale);
//
//   std::vector<float> h_output(npix);
//   CHECK_CUDA(cudaMemcpyAsync(h_output.data(), slice.data_handle(), npix * sizeof(float),
//                              cudaMemcpyDeviceToHost, stream_res.cuda_stream));
//   stream_res.sync();
//
//   const float* expected = expected_npy.as_float32();
//   for (int i = 0; i < npix; i++) {
//     ASSERT_EQ(h_output[i], expected[i]) << "Mismatch at pixel " << i;
//   }
//
//   resources.free_async(d_src, stream_res);
//   stream_res.sync();
// }
