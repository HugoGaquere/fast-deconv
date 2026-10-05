#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cub/cub.cuh>
#include <emu/submdspan.hpp>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/gaussian_convolution.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <fast_deconv/morphology/dilation.hpp>
#include <fast_deconv/morphology/roi.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fast_deconv/util/dump.hpp>
#include <limits>
#include <numbers>
#include <vector>

namespace fast_deconv::kernel {

__global__ void build_mask_per_scale_kernel(const int2* coords, const int* scales, bool* out_mask_per_scale,
                                            int n_coords, int inner_scale_stride, int inter_scales_stride)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_coords) return;
  const std::int64_t idx = static_cast<std::int64_t>(scales[tid]) * inter_scales_stride +
                           coords[tid].x * inner_scale_stride + coords[tid].y;  // int2: x is the row, y the column
  out_mask_per_scale[idx] = true;
}

// Threshold a real image into a bool FWHM mask.
__global__ void threshold_fwhm_kernel(const float* data, float threshold, bool* out, int n)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  out[tid] = data[tid] > threshold;
}

// Negate each scale slice of mask_per_scale and OR external_mask into it.
// After this pass: mask_per_scale[s, i] = (!is_near_component[s, i]) || external_mask[i].
__global__ void finalize_mask_kernel(bool* mask_per_scale, const bool* external_mask, int scale_npix, int n_scales)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= scale_npix) return;
  const bool ext = external_mask[tid];
  bool* slice = mask_per_scale;
  for (int s = 0; s < n_scales; s++, slice += scale_npix) slice[tid] = (!slice[tid]) || ext;
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::common {

void build_auto_mask(const core::exec_ctx& ctx, const std::vector<index2d>& coords, const std::vector<int>& scales,
                     core::span3d<float> central_facet_psfs, core::span1d<const float> weights_freq,
                     const std::vector<float>& scale_sigmas, core::span2d<bool> external_mask,
                     core::span3d<bool> mask_per_scale)
{
  assert(mask_per_scale.is_exhaustive());
  assert(external_mask.is_exhaustive());
  assert(central_facet_psfs.is_exhaustive());
  assert(coords.size() == scales.size());
  assert(weights_freq.extent(0) == central_facet_psfs.extent(0));
  assert(external_mask.extent(0) == mask_per_scale.extent(1) && external_mask.extent(1) == mask_per_scale.extent(2));

  const int n_coords = static_cast<int>(coords.size());
  const int n_scales = mask_per_scale.extent(0);
  const int dirty_nrow = mask_per_scale.extent(1);
  const int dirty_ncol = mask_per_scale.extent(2);
  const int dirty_npix = dirty_nrow * dirty_ncol;
  const int n_freq = central_facet_psfs.extent(0);
  const int psf_nrow = central_facet_psfs.extent(1);
  const int psf_ncol = central_facet_psfs.extent(2);
  const int psf_npix = psf_nrow * psf_ncol;
  const cudaStream_t cuda_stream = ctx.cuda_stream;

  // ---- 1. Zero the output and stamp peaks ----
  CHECK_CUDA(cudaMemsetAsync(mask_per_scale.data_handle(), 0, mask_per_scale.size() * sizeof(bool), cuda_stream));

  if (n_coords > 0) {
    std::vector<int2> h_coords(n_coords);
    for (int i = 0; i < n_coords; i++) h_coords[i] = make_int2(coords[i].row, coords[i].col);

    auto d_coords = ctx.alloc_mdcontainer_async<int2>(n_coords);
    auto d_scales = ctx.alloc_mdcontainer_async<int>(n_coords);
    CHECK_CUDA(cudaMemcpyAsync(d_coords.data_handle(), h_coords.data(), n_coords * sizeof(int2), cudaMemcpyHostToDevice,
                               cuda_stream));
    CHECK_CUDA(cudaMemcpyAsync(d_scales.data_handle(), scales.data(), n_coords * sizeof(int), cudaMemcpyHostToDevice,
                               cuda_stream));

    dim3 block(256);
    dim3 grid(CEIL_DIV(n_coords, block.x));
    kernel::build_mask_per_scale_kernel<<<grid, block, 0, cuda_stream>>>(
        d_coords.data_handle(), d_scales.data_handle(), mask_per_scale.data_handle(), n_coords, dirty_ncol, dirty_npix);
  }

  // ---- 2. One batched forward transform of every channel's PSF, reused by every scale ----
  // Wrap-free for every scale's sigma * sqrt(2).
  const linalg::gaussian_convolution_ctx conv(ctx, /*batch=*/n_freq, psf_nrow, psf_ncol,
                                              linalg::max_gaussian_reach(scale_sigmas, std::numbers::sqrt2_v<float>));
  auto spectrum = conv.make_spectrum();
  conv.forward(central_facet_psfs, spectrum);

  // ---- 3. Per-scale: conv2 -> weighted mean -> FWHM -> dilate ----
  auto conv2_cropped = ctx.alloc_mdcontainer_async<float>(n_freq, psf_nrow, psf_ncol);
  auto conv2_psf = ctx.alloc_mdcontainer_async<float>(psf_nrow, psf_ncol);
  auto fwhm_mask = ctx.alloc_mdcontainer_async<bool>(psf_nrow, psf_ncol);
  auto dilation_out = ctx.alloc_mdcontainer_async<bool>(dirty_nrow, dirty_ncol);

  for (int i = 0; i < n_scales; i++) {
    // 3a. psf ** G_s ** G_s, one convolution with G(sigma * sqrt(2)) since G(sigma)^2 = G(sigma * sqrt(2)).
    conv.convolve(spectrum, scale_sigmas.at(i) * std::numbers::sqrt2_v<float>, conv2_cropped);

    // 3b. Weighted mean across channels -> 2D conv2_psf
    linalg::weighted_sum_async(ctx, conv2_cropped, weights_freq, conv2_psf);

    // 3c. Max-reduce conv2_psf (synchronizes the stream)
    auto max_iter =
        thrust::max_element(thrust::cuda::par.on(cuda_stream), thrust::device_pointer_cast(conv2_psf.data_handle()),
                            thrust::device_pointer_cast(conv2_psf.data_handle() + psf_npix));
    float psf_max = 0.0f;
    CHECK_CUDA(cudaMemcpyAsync(&psf_max, thrust::raw_pointer_cast(max_iter), sizeof(float), cudaMemcpyDeviceToHost,
                               cuda_stream));
    ctx.wait();

    // 3d. FWHM threshold: bool out = (conv2_psf > psf_max / 2)
    kernel::threshold_fwhm_kernel<<<CEIL_DIV(psf_npix, 256), 256, 0, cuda_stream>>>(
        conv2_psf.data_handle(), psf_max * 0.5f, fwhm_mask.data_handle(), psf_npix);

    // 3e. Bounding box of FWHM (synchronous host-side reduction)
    const roi structure_roi = morphology::compute_mask_roi(ctx, fwhm_mask);

    // 3f. Dilate mask_per_scale[i] using FWHM as structuring element
    // binary_dilation only writes out[tid]=true on matches; zero the buffer first.
    CHECK_CUDA(cudaMemsetAsync(dilation_out.data_handle(), 0, dirty_npix * sizeof(bool), cuda_stream));
    core::span2d<bool> current_mask = emu::submdspan(mask_per_scale, i);
    morphology::binary_dilation(ctx, current_mask, fwhm_mask, structure_roi, dilation_out);

    // 3g. Copy dilation result back into mask_per_scale[i]
    CHECK_CUDA(cudaMemcpyAsync(current_mask.data_handle(), dilation_out.data_handle(), dirty_npix * sizeof(bool),
                               cudaMemcpyDeviceToDevice, cuda_stream));
  }

  // ---- 4. Negate to "true=masked" convention and OR external_mask into every scale slice ----
  kernel::finalize_mask_kernel<<<CEIL_DIV(dirty_npix, 256), 256, 0, cuda_stream>>>(
      mask_per_scale.data_handle(), external_mask.data_handle(), dirty_npix, n_scales);
}
}  // namespace fast_deconv::common
