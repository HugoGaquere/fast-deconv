#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <cassert>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <fast_deconv/morphology/dilation.hpp>
#include <fast_deconv/morphology/roi.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <limits>
#include <vector>

namespace fast_deconv::kernel {

__global__ void mask_and_abs_kernel(float* data, const bool* mask, float fill_value, bool abs, int n_per_batch,
                                    int data_batch_stride, int mask_batch_stride)
{
  float* d = data + blockIdx.y * data_batch_stride;
  const bool* m = mask + blockIdx.y * mask_batch_stride;
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_per_batch; i += stride) {
    if (m[i])
      d[i] = fill_value;
    else if (abs)
      d[i] = fabsf(d[i]);
  }
}

__global__ void mask_less_than_threshold_kernel(float* data, float threshold, float fill_value, int n_per_batch,
                                                int batch_stride)
{
  float* d = data + blockIdx.y * batch_stride;
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_per_batch; i += stride) {
    if (data[i] < threshold) d[i] = fill_value;
  }
}

__global__ void build_mask_per_scale_kernel(const int2* coords, const int* scales, bool* out_mask_per_scale,
                                            int n_coords, int inner_scale_stride, int inter_scales_stride)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_coords) return;
  const int idx = scales[tid] * inter_scales_stride + coords[tid].x * inner_scale_stride + coords[tid].y;
  out_mask_per_scale[idx] = true;
}

// Batched G^2-only variant: freq_out[b,i] = freq_psf[b,i] * gaussian[i]^2 * norm.
__global__ void multiply_psf_gauss_sq_batched_kernel(const complex_type* freq_psf, const float* gaussian,
                                                     complex_type* freq_out, int freq_total, int n_batch, float norm)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= freq_total) return;
  const float g = gaussian[tid];
  const float g2_norm = g * g * norm;
  for (int b = 0; b < n_batch; b++) {
    const int idx = b * freq_total + tid;
    const complex_type val = freq_psf[idx];
    freq_out[idx] = {val.x * g2_norm, val.y * g2_norm};
  }
}

// Threshold a real image into a bool FWHM mask.
__global__ void threshold_fwhm_kernel(const float* data, float threshold, bool* out, int n)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  out[tid] = data[tid] > threshold;
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::common {

void mask_and_abs_async(const core::stream_resources& stream_res, core::device_span2d<float> data,
                        core::device_span2d<bool> mask, float fill_value, bool abs)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extent(0) == mask.extent(0) && data.extent(1) == mask.extent(1));
  const int n = static_cast<int>(data.size());
  dim3 grid(CEIL_DIV(n, 256), 1);
  kernel::mask_and_abs_kernel<<<grid, 256, 0, stream_res.cuda_stream>>>(data.data_handle(), mask.data_handle(),
                                                                        fill_value, abs, n, 0, 0);
}

void mask_and_abs_async(const core::stream_resources& stream_res, core::device_span3d<float> data,
                        core::device_span2d<bool> mask, float fill_value, bool abs)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extent(1) == mask.extent(0) && data.extent(2) == mask.extent(1));
  const int n_per_batch = static_cast<int>(mask.size());
  const int nbatch = static_cast<int>(data.extent(0));
  const int batch_stride = static_cast<int>(data.stride(0));
  dim3 grid(CEIL_DIV(n_per_batch, 256), nbatch);
  kernel::mask_and_abs_kernel<<<grid, 256, 0, stream_res.cuda_stream>>>(data.data_handle(), mask.data_handle(),
                                                                        fill_value, abs, n_per_batch, batch_stride, 0);
}

void mask_and_abs_async(const core::stream_resources& stream_res, core::device_span3d<float> data,
                        core::device_span3d<bool> mask, float fill_value, bool abs)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extents() == mask.extents());
  const int n_per_batch = static_cast<int>(mask.size());
  const int nbatch = static_cast<int>(data.extent(0));
  const int batch_stride = static_cast<int>(data.stride(0));
  dim3 grid(CEIL_DIV(n_per_batch, 256), nbatch);
  kernel::mask_and_abs_kernel<<<grid, 256, 0, stream_res.cuda_stream>>>(
      data.data_handle(), mask.data_handle(), fill_value, abs, n_per_batch, batch_stride, batch_stride);
}
void mask_less_than_threshold(const core::stream_resources& stream_res, core::device_span2d<float> data,
                              float threshold, float fill_value)
{
  assert(data.is_exhaustive());
  const int n = static_cast<int>(data.size());
  dim3 grid(CEIL_DIV(n, 256), 1);
  kernel::mask_less_than_threshold_kernel<<<grid, 256, 0, stream_res.cuda_stream>>>(data.data_handle(), threshold,
                                                                                    fill_value, n, 0);
}

void build_independant_scale_mask(const core::resources& resources, const core::stream_resources& stream,
                                  const std::vector<std::pair<int, int>>& coords, const std::vector<int>& scales,
                                  core::device_span3d<float> central_facet_psfs,
                                  core::device_vect<float> weights_freq, core::device_vect<float> scale_sigmas,
                                  float fft_padding, core::device_span3d<bool> mask_per_scale)
{
  assert(mask_per_scale.is_exhaustive());
  assert(central_facet_psfs.is_exhaustive());
  assert(coords.size() == scales.size());
  assert(static_cast<int>(weights_freq.extent(0)) == static_cast<int>(central_facet_psfs.extent(0)));

  const int n_coords = static_cast<int>(coords.size());
  const int n_scales = static_cast<int>(mask_per_scale.extent(0));
  const int dirty_nrow = static_cast<int>(mask_per_scale.extent(1));
  const int dirty_ncol = static_cast<int>(mask_per_scale.extent(2));
  const int dirty_npix = dirty_nrow * dirty_ncol;
  const int n_freq = static_cast<int>(central_facet_psfs.extent(0));
  const int psf_nrow = static_cast<int>(central_facet_psfs.extent(1));
  const int psf_ncol = static_cast<int>(central_facet_psfs.extent(2));
  const int psf_npix = psf_nrow * psf_ncol;
  const cudaStream_t cuda_stream = stream.cuda_stream;

  // ---- 1. Zero the output and stamp peaks ----
  CHECK_CUDA(cudaMemsetAsync(mask_per_scale.data_handle(), 0, mask_per_scale.size() * sizeof(bool), cuda_stream));

  if (n_coords > 0) {
    std::vector<int2> h_coords(n_coords);
    for (int i = 0; i < n_coords; i++) h_coords[i] = make_int2(coords[i].first, coords[i].second);

    int2* d_coords = stream.alloc_async<int2>(n_coords);
    int* d_scales = stream.alloc_async<int>(n_coords);
    CHECK_CUDA(cudaMemcpyAsync(d_coords, h_coords.data(), n_coords * sizeof(int2), cudaMemcpyHostToDevice, cuda_stream));
    CHECK_CUDA(cudaMemcpyAsync(d_scales, scales.data(), n_coords * sizeof(int), cudaMemcpyHostToDevice, cuda_stream));

    dim3 block(256);
    dim3 grid(CEIL_DIV(n_coords, block.x));
    kernel::build_mask_per_scale_kernel<<<grid, block, 0, cuda_stream>>>(d_coords, d_scales, mask_per_scale.data_handle(),
                                                                         n_coords, dirty_ncol, dirty_npix);

    stream.free_async(d_coords);
    stream.free_async(d_scales);
  }

  // ---- 2. Build a PSF-sized convolve_ctx with batched plans over n_freq (1 R2C + 1 C2R) ----
  linalg::convolve_ctx ctx(psf_nrow, psf_ncol, /*plan_batch=*/n_freq, /*n_backward_plans=*/1, fft_padding);
  void* fft_work = resources.alloc_async<void>(ctx.required_work_size(), stream);
  ctx.set_work_area(fft_work);
  ctx.set_stream(cuda_stream);

  const int padded_total = ctx.padded_nrow * ctx.padded_ncol;
  const int freq_total = ctx.freq_nrow * ctx.freq_ncol;

  // ---- 3. Generate per-scale Gaussian kernels at PSF FFT size ----
  float* gauss_kernels_ptr = resources.alloc_async<float>(static_cast<uint64_t>(n_scales) * freq_total, stream);
  core::device_span3d<float> gauss_kernels(gauss_kernels_ptr, n_scales, ctx.freq_nrow, ctx.freq_ncol);
  scale::make_gaussian_kernels_async(stream, scale_sigmas, ctx.padded_ncol, gauss_kernels);

  // ---- 4. Batched pad+ifftshift and R2C of all frequency PSFs (once, reused across scales) ----
  float* padded_psf = resources.alloc_async<float>(static_cast<uint64_t>(n_freq) * padded_total, stream);
  complex_type* freq_psf = resources.alloc_async<complex_type>(static_cast<uint64_t>(n_freq) * freq_total, stream);

  linalg::pad_ifftshift_batched(central_facet_psfs.data_handle(), padded_psf, psf_nrow, psf_ncol, ctx.padded_nrow,
                                ctx.padded_ncol, ctx.padding_nrow, ctx.padding_ncol, n_freq, cuda_stream);
  CUFFT_CALL(cufftExecR2C(ctx.plans_forward[0], padded_psf, freq_psf));

  // ---- 5. Per-scale: multiply (batched) -> C2R (batched) -> crop (batched) -> weighted mean -> ... ----
  complex_type* freq_conv2 = resources.alloc_async<complex_type>(static_cast<uint64_t>(n_freq) * freq_total, stream);
  float* padded_conv2 = resources.alloc_async<float>(static_cast<uint64_t>(n_freq) * padded_total, stream);
  float* conv2_cropped_ptr = resources.alloc_async<float>(static_cast<uint64_t>(n_freq) * psf_npix, stream);
  core::device_span3d<float> conv2_cropped(conv2_cropped_ptr, n_freq, psf_nrow, psf_ncol);
  float* conv2_psf = resources.alloc_async<float>(psf_npix, stream);
  core::device_span2d<float> conv2_psf_view(conv2_psf, psf_nrow, psf_ncol);
  bool* fwhm_mask = resources.alloc_async<bool>(psf_npix, stream);
  bool* dilation_out = resources.alloc_async<bool>(dirty_npix, stream);

  const float norm = 1.0f / static_cast<float>(padded_total);

  for (int i = 0; i < n_scales; i++) {
    // 5a. Batched multiply by G^2 in freq domain (with norm)
    kernel::multiply_psf_gauss_sq_batched_kernel<<<CEIL_DIV(freq_total, 256), 256, 0, cuda_stream>>>(
        freq_psf, gauss_kernels_ptr + i * freq_total, freq_conv2, freq_total, n_freq, norm);

    // 5b. Batched C2R back to space
    CUFFT_CALL(cufftExecC2R(ctx.plans_backward[0], freq_conv2, padded_conv2));

    // 5c. Batched fftshift + crop to PSF size
    linalg::fftshift_crop(padded_conv2, conv2_cropped_ptr, psf_nrow, psf_ncol, ctx.padded_nrow, ctx.padded_ncol,
                          ctx.padding_nrow, ctx.padding_ncol, n_freq, cuda_stream);

    // 5d. Weighted mean across channels -> 2D conv2_psf
    linalg::weighted_sum_async(stream, conv2_cropped, weights_freq, conv2_psf_view);

    // 5e. Max-reduce conv2_psf (synchronizes the stream)
    auto max_iter = thrust::max_element(thrust::cuda::par.on(cuda_stream), thrust::device_pointer_cast(conv2_psf),
                                        thrust::device_pointer_cast(conv2_psf + psf_npix));
    float psf_max = 0.0f;
    CHECK_CUDA(cudaMemcpyAsync(&psf_max, thrust::raw_pointer_cast(max_iter), sizeof(float), cudaMemcpyDeviceToHost,
                               cuda_stream));
    stream.sync();

    // 5f. FWHM threshold: bool out = (conv2_psf > psf_max / 2)
    kernel::threshold_fwhm_kernel<<<CEIL_DIV(psf_npix, 256), 256, 0, cuda_stream>>>(conv2_psf, psf_max * 0.5f, fwhm_mask,
                                                                                    psf_npix);

    // 5g. Bounding box of FWHM (synchronous host-side reduction)
    core::device_span2d<bool> fwhm_view(fwhm_mask, psf_nrow, psf_ncol);
    morphology::roi structure_roi = morphology::compute_mask_roi(stream, fwhm_view);

    // Convert inclusive bounds (compute_mask_roi) to exclusive (binary_dilation expects exclusive max).
    structure_roi.xmax += 1;
    structure_roi.ymax += 1;

    // 5h. Dilate mask_per_scale[i] using FWHM as structuring element
    // binary_dilation only writes out[tid]=true on matches; zero the buffer first.
    CHECK_CUDA(cudaMemsetAsync(dilation_out, 0, dirty_npix * sizeof(bool), cuda_stream));
    auto current_mask = core::slice_leading(mask_per_scale, i);
    core::device_span2d<bool> dilation_out_view(dilation_out, dirty_nrow, dirty_ncol);
    morphology::binary_dilation(stream, current_mask, fwhm_view, structure_roi, dilation_out_view);

    // 5i. Copy dilation result back into mask_per_scale[i]
    CHECK_CUDA(cudaMemcpyAsync(current_mask.data_handle(), dilation_out, dirty_npix * sizeof(bool),
                               cudaMemcpyDeviceToDevice, cuda_stream));
  }

  // ---- 6. Cleanup ----
  resources.free_async(dilation_out, stream);
  resources.free_async(fwhm_mask, stream);
  resources.free_async(conv2_psf, stream);
  resources.free_async(conv2_cropped_ptr, stream);
  resources.free_async(padded_conv2, stream);
  resources.free_async(freq_conv2, stream);
  resources.free_async(freq_psf, stream);
  resources.free_async(padded_psf, stream);
  resources.free_async(gauss_kernels_ptr, stream);
  resources.free_async(fft_work, stream);
}
}  // namespace fast_deconv::common
