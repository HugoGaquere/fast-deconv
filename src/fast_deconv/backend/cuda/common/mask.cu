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
#include <iterator>
#include <limits>
#include <numbers>
#include <vector>

namespace fast_deconv::kernel {

__global__ void build_mask_per_scale_kernel(const std::int64_t* offsets, bool* out_mask_per_scale,
                                            std::int64_t n_coords)
{
  const std::int64_t tid = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (tid >= n_coords) return;
  out_mask_per_scale[offsets[tid]] = true;
}

// Threshold a real image into a bool FWHM mask.
__global__ void threshold_fwhm_kernel(const float* data, float threshold, bool* out, std::int64_t n)
{
  const std::int64_t tid = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  out[tid] = data[tid] > threshold;
}

// Negate each scale slice of mask_per_scale and OR external_mask into it.
// After this pass: mask_per_scale[s, i] = (!is_near_component[s, i]) || external_mask[i].
__global__ void finalize_mask_kernel(bool* mask_per_scale, const bool* external_mask, std::int64_t scale_npix,
                                     std::int64_t n_scales)
{
  const std::int64_t tid = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (tid >= scale_npix) return;
  const bool ext = external_mask[tid];
  bool* slice = mask_per_scale;
  for (std::int64_t s = 0; s < n_scales; s++, slice += scale_npix) slice[tid] = (!slice[tid]) || ext;
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::common {

void build_auto_mask(const core::exec_ctx& ctx, const std::vector<index2d>& coords, const std::vector<int>& scales,
                     core::span3d<const float> central_facet_psfs, core::span1d<const float> weights_freq,
                     const std::vector<float>& scale_sigmas, core::span2d<const bool> external_mask,
                     core::span3d<bool> mask_per_scale)
{
  assert(mask_per_scale.is_exhaustive());
  assert(external_mask.is_exhaustive());
  assert(central_facet_psfs.is_exhaustive());
  assert(coords.size() == scales.size());
  assert(weights_freq.extent(0) == central_facet_psfs.extent(0));
  assert(external_mask.extent(0) == mask_per_scale.extent(1) && external_mask.extent(1) == mask_per_scale.extent(2));

  const std::int64_t n_scales = mask_per_scale.extent(0);
  const std::int64_t dirty_nrow = mask_per_scale.extent(1);
  const std::int64_t dirty_ncol = mask_per_scale.extent(2);
  const std::int64_t dirty_npix = mask_per_scale.mapping().stride(0);
  const std::int64_t psf_nrow = central_facet_psfs.extent(1);
  const std::int64_t psf_ncol = central_facet_psfs.extent(2);
  const std::int64_t psf_npix = central_facet_psfs.mapping().stride(0);
  const cudaStream_t cuda_stream = ctx.cuda_stream;

  // ---- 1. Zero the output and stamp peaks ----
  CHECK_CUDA(cudaMemsetAsync(mask_per_scale.data_handle(), 0, mask_per_scale.size() * sizeof(bool), cuda_stream));

  if (!coords.empty()) {
    // Flat offsets on the host: the kernel then needs neither the coordinates nor the strides.
    std::vector<std::int64_t> h_offsets;
    h_offsets.reserve(coords.size());
    for (std::size_t i = 0; i < coords.size(); i++)
      h_offsets.push_back(mask_per_scale.mapping()(scales.at(i), coords.at(i).row, coords.at(i).col));

    const std::int64_t n_coords = std::ssize(h_offsets);
    auto d_offsets = ctx.alloc_mdcontainer_async<std::int64_t>(n_coords);
    ctx.copy(d_offsets.data_handle(), h_offsets.data(), n_coords);

    dim3 block(256);
    dim3 grid(core::narrow<unsigned>(core::ceil_div(n_coords, block.x)));
    kernel::build_mask_per_scale_kernel<<<grid, block, 0, cuda_stream>>>(d_offsets.data_handle(),
                                                                         mask_per_scale.data_handle(), n_coords);
  }

  // ---- 2. Weighted mean of the channels' PSFs, transformed once and reused by every scale ----
  // Convolution is linear: sum_f w_f (G * psf_f) = G * (sum_f w_f psf_f), so one plane instead of n_freq.
  auto psf_mean = ctx.alloc_mdcontainer_async<float>(psf_nrow, psf_ncol);
  linalg::weighted_sum_async(ctx, central_facet_psfs, weights_freq, psf_mean);
  // Wrap-free for every scale's sigma * sqrt(2).
  const linalg::gaussian_convolution_ctx conv(ctx, /*batch=*/1, psf_nrow, psf_ncol, scale_sigmas,
                                              std::numbers::sqrt2_v<float>);
  auto spectrum = conv.make_spectrum();
  conv.forward(core::span3d<const float>(psf_mean.data_handle(), 1, psf_nrow, psf_ncol), spectrum);

  // ---- 3. Per-scale: conv2 -> FWHM -> dilate ----
  auto conv2_psf = ctx.alloc_mdcontainer_async<float>(psf_nrow, psf_ncol);
  auto fwhm_mask = ctx.alloc_mdcontainer_async<bool>(psf_nrow, psf_ncol);
  auto dilation_out = ctx.alloc_mdcontainer_async<bool>(dirty_nrow, dirty_ncol);

  for (std::int64_t i = 0; i < n_scales; i++) {
    // 3a. psf_mean ** G_s ** G_s, one convolution with G(sigma * sqrt(2)) since G(sigma)^2 = G(sigma * sqrt(2)).
    conv.convolve(spectrum, scale_sigmas.at(core::narrow<std::size_t>(i)) * std::numbers::sqrt2_v<float>,
                  core::span3d<float>(conv2_psf.data_handle(), 1, psf_nrow, psf_ncol));

    // 3b. Max-reduce conv2_psf (synchronizes the stream)
    auto max_iter =
        thrust::max_element(thrust::cuda::par.on(cuda_stream), thrust::device_pointer_cast(conv2_psf.data_handle()),
                            thrust::device_pointer_cast(conv2_psf.data_handle() + psf_npix));
    float psf_max = 0.0f;
    CHECK_CUDA(cudaMemcpyAsync(&psf_max, thrust::raw_pointer_cast(max_iter), sizeof(float), cudaMemcpyDeviceToHost,
                               cuda_stream));
    ctx.wait();

    // 3c. FWHM threshold: bool out = (conv2_psf > psf_max / 2)
    kernel::threshold_fwhm_kernel<<<core::narrow<unsigned>(core::ceil_div(psf_npix, 256)), 256, 0, cuda_stream>>>(
        conv2_psf.data_handle(), psf_max * 0.5f, fwhm_mask.data_handle(), psf_npix);

    // 3d. Bounding box of FWHM (synchronous host-side reduction)
    const roi structure_roi = morphology::compute_mask_roi(ctx, fwhm_mask);

    // 3e. Dilate mask_per_scale[i] using FWHM as structuring element
    // binary_dilation only writes out[tid]=true on matches; zero the buffer first.
    CHECK_CUDA(cudaMemsetAsync(dilation_out.data_handle(), 0, dilation_out.size() * sizeof(bool), cuda_stream));
    core::span2d<bool> current_mask = emu::submdspan(mask_per_scale, i);
    morphology::binary_dilation(ctx, current_mask, fwhm_mask, structure_roi, dilation_out);

    ctx.copy(current_mask.data_handle(), dilation_out.data_handle(), dirty_npix);
  }

  // ---- 4. Negate to "true=masked" convention and OR external_mask into every scale slice ----
  kernel::finalize_mask_kernel<<<core::narrow<unsigned>(core::ceil_div(dirty_npix, 256)), 256, 0, cuda_stream>>>(
      mask_per_scale.data_handle(), external_mask.data_handle(), dirty_npix, n_scales);
}
}  // namespace fast_deconv::common
