#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cub/cub.cuh>
#include <emu/submdspan.hpp>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <fast_deconv/morphology/dilation.hpp>
#include <fast_deconv/morphology/roi.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fast_deconv/util/dump.hpp>
#include <limits>
#include <vector>

namespace fast_deconv::kernel {

// Block/thread tiling for the elementwise mask kernels. ITEMS_PER_THREAD=4 makes
// the VECTORIZE algorithms emit 128-bit (float4 / uchar4) transactions; CUB checks
// pointer alignment at runtime and falls back to scalar guarded loads otherwise,
// and the partial-tile API bounds-guards the trailing tile. Delegating the vector
// width to CUB keeps this correct/tuned as the target architecture changes.
namespace detail {
constexpr int kMaskBlock = 256;
constexpr int kMaskItemsPerThread = 4;
constexpr int kMaskTile = kMaskBlock * kMaskItemsPerThread;
using LoadFloat = cub::BlockLoad<float, kMaskBlock, kMaskItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
using StoreFloat = cub::BlockStore<float, kMaskBlock, kMaskItemsPerThread, cub::BLOCK_STORE_VECTORIZE>;
using LoadMask = cub::BlockLoad<unsigned char, kMaskBlock, kMaskItemsPerThread, cub::BLOCK_LOAD_VECTORIZE>;
}  // namespace detail

__global__ void mask_and_abs_kernel(float* data, const bool* mask, float fill_value, bool abs, int n_per_batch,
                                    std::int64_t data_batch_stride, std::int64_t mask_batch_stride)
{
  float* d = data + blockIdx.y * data_batch_stride;
  const unsigned char* m = reinterpret_cast<const unsigned char*>(mask) + blockIdx.y * mask_batch_stride;
  const int tile = detail::kMaskTile;
  for (int base = blockIdx.x * tile; base < n_per_batch; base += gridDim.x * tile) {
    const int valid = min(tile, n_per_batch - base);
    float dv[detail::kMaskItemsPerThread];
    unsigned char mv[detail::kMaskItemsPerThread];
    detail::LoadFloat().Load(d + base, dv, valid);
    detail::LoadMask().Load(m + base, mv, valid);
#pragma unroll
    for (int j = 0; j < detail::kMaskItemsPerThread; ++j) dv[j] = mv[j] ? fill_value : (abs ? fabsf(dv[j]) : dv[j]);
    detail::StoreFloat().Store(d + base, dv, valid);
  }
}

__global__ void mask_less_than_threshold_kernel(float* data, float threshold, float fill_value, int n_per_batch,
                                                std::int64_t batch_stride)
{
  float* d = data + blockIdx.y * batch_stride;
  const int tile = detail::kMaskTile;
  for (int base = blockIdx.x * tile; base < n_per_batch; base += gridDim.x * tile) {
    const int valid = min(tile, n_per_batch - base);
    float dv[detail::kMaskItemsPerThread];
    detail::LoadFloat().Load(d + base, dv, valid);
#pragma unroll
    for (int j = 0; j < detail::kMaskItemsPerThread; ++j)
      if (dv[j] < threshold) dv[j] = fill_value;
    detail::StoreFloat().Store(d + base, dv, valid);
  }
}

__global__ void build_mask_per_scale_kernel(const int2* coords, const int* scales, bool* out_mask_per_scale,
                                            int n_coords, int inner_scale_stride, int inter_scales_stride)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_coords) return;
  const std::int64_t idx =
      static_cast<std::int64_t>(scales[tid]) * inter_scales_stride + coords[tid].x * inner_scale_stride + coords[tid].y;
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
  const complex_type* in = freq_psf + tid;
  complex_type* out = freq_out + tid;
  for (int b = 0; b < n_batch; b++, in += freq_total, out += freq_total) {
    const complex_type val = *in;
    *out = {val.x * g2_norm, val.y * g2_norm};
  }
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

void mask_and_abs_async(const core::stream_resources& stream_res, core::device_span2d<float> data,
                        core::device_span2d<bool> mask, float fill_value, bool abs)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extent(0) == mask.extent(0) && data.extent(1) == mask.extent(1));
  const int n = static_cast<int>(data.size());
  dim3 grid(CEIL_DIV(n, kernel::detail::kMaskTile), 1);
  kernel::mask_and_abs_kernel<<<grid, kernel::detail::kMaskBlock, 0, stream_res.cuda_stream>>>(
      data.data_handle(), mask.data_handle(), fill_value, abs, n, 0, 0);
}

void mask_and_abs_async(const core::stream_resources& stream_res, core::device_span3d<float> data,
                        core::device_span2d<bool> mask, float fill_value, bool abs)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extent(1) == mask.extent(0) && data.extent(2) == mask.extent(1));
  const int n_per_batch = static_cast<int>(mask.size());
  const int nbatch = data.extent(0);
  const std::int64_t batch_stride = data.stride(0);
  dim3 grid(CEIL_DIV(n_per_batch, kernel::detail::kMaskTile), nbatch);
  kernel::mask_and_abs_kernel<<<grid, kernel::detail::kMaskBlock, 0, stream_res.cuda_stream>>>(
      data.data_handle(), mask.data_handle(), fill_value, abs, n_per_batch, batch_stride, 0);
}

void mask_and_abs_async(const core::stream_resources& stream_res, core::device_span3d<float> data,
                        core::device_span3d<bool> mask, float fill_value, bool abs)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extents() == mask.extents());
  const int n_per_batch = static_cast<int>(data.extent(1) * data.extent(2));
  const int nbatch = data.extent(0);
  const std::int64_t batch_stride = data.stride(0);
  dim3 grid(CEIL_DIV(n_per_batch, kernel::detail::kMaskTile), nbatch);
  kernel::mask_and_abs_kernel<<<grid, kernel::detail::kMaskBlock, 0, stream_res.cuda_stream>>>(
      data.data_handle(), mask.data_handle(), fill_value, abs, n_per_batch, batch_stride, batch_stride);
}
void mask_less_than_threshold(const core::stream_resources& stream_res, core::device_span2d<float> data,
                              float threshold, float fill_value)
{
  assert(data.is_exhaustive());
  const int n = static_cast<int>(data.size());
  dim3 grid(CEIL_DIV(n, kernel::detail::kMaskTile), 1);
  kernel::mask_less_than_threshold_kernel<<<grid, kernel::detail::kMaskBlock, 0, stream_res.cuda_stream>>>(
      data.data_handle(), threshold, fill_value, n, 0);
}

void build_auto_mask(const core::stream_resources& stream, const std::vector<std::pair<int, int>>& coords,
                     const std::vector<int>& scales, core::device_span3d<float> central_facet_psfs,
                     core::device_vect<float> weights_freq, core::device_vect<float> scale_sigmas, float fft_padding,
                     core::device_span2d<bool> external_mask, core::device_span3d<bool> mask_per_scale)
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
  const cudaStream_t cuda_stream = stream.cuda_stream;

  // ---- 1. Zero the output and stamp peaks ----
  CHECK_CUDA(cudaMemsetAsync(mask_per_scale.data_handle(), 0, mask_per_scale.size() * sizeof(bool), cuda_stream));

  if (n_coords > 0) {
    std::vector<int2> h_coords(n_coords);
    for (int i = 0; i < n_coords; i++) h_coords[i] = make_int2(coords[i].first, coords[i].second);

    auto d_coords = stream.alloc_mdcontainer_async<int2>(n_coords);
    auto d_scales = stream.alloc_mdcontainer_async<int>(n_coords);
    CHECK_CUDA(cudaMemcpyAsync(d_coords.data_handle(), h_coords.data(), n_coords * sizeof(int2), cudaMemcpyHostToDevice,
                               cuda_stream));
    CHECK_CUDA(cudaMemcpyAsync(d_scales.data_handle(), scales.data(), n_coords * sizeof(int), cudaMemcpyHostToDevice,
                               cuda_stream));

    dim3 block(256);
    dim3 grid(CEIL_DIV(n_coords, block.x));
    kernel::build_mask_per_scale_kernel<<<grid, block, 0, cuda_stream>>>(
        d_coords.data_handle(), d_scales.data_handle(), mask_per_scale.data_handle(), n_coords, dirty_ncol, dirty_npix);
  }

  // ---- 2. Build a PSF-sized convolve_ctx with batched plans over n_freq (1 R2C + 1 C2R) ----
  // ctx runs its plans on stream.cuda_stream (== cuda_stream below). Its plans have
  // auto-allocation disabled, so a caller-owned work area must be bound before any exec.
  linalg::convolve_ctx ctx(stream, psf_nrow, psf_ncol, /*forward_batch=*/n_freq, /*backward_batch=*/n_freq,
                           /*n_backward_plans=*/1, fft_padding);

  core::device_ptr<std::byte> fft_work_area;
  if (ctx.required_work_size() > 0) fft_work_area = stream.alloc_ptr_async<std::byte>(ctx.required_work_size());
  ctx.bind_work_area(fft_work_area.get());

  const int padded_total = ctx.padded_nrow * ctx.padded_ncol;
  const int freq_total = ctx.freq_nrow * ctx.freq_ncol;

  // ---- 3. Generate per-scale Gaussian kernels at PSF FFT size ----
  auto gauss_kernels = stream.alloc_mdcontainer_async<float>(n_scales, ctx.freq_nrow, ctx.freq_ncol);
  scale::make_gaussian_kernels_async(stream, scale_sigmas, ctx.padded_ncol, gauss_kernels);

  // ---- 4. Batched pad+ifftshift and R2C of all frequency PSFs (once, reused across scales) ----
  auto padded_psf = stream.alloc_mdcontainer_async<float>(n_freq, padded_total);
  auto freq_psf = stream.alloc_mdcontainer_async<complex_type>(n_freq, freq_total);

  linalg::pad_ifftshift_batched(central_facet_psfs.data_handle(), padded_psf.data_handle(), psf_nrow, psf_ncol,
                                ctx.padded_nrow, ctx.padded_ncol, ctx.padding_nrow, ctx.padding_ncol, n_freq,
                                cuda_stream);
  CUFFT_CALL(cufftExecR2C(ctx.plans_forward[0], padded_psf.data_handle(), freq_psf.data_handle()));

  // ---- 5. Per-scale: multiply (batched) -> C2R (batched) -> crop (batched) -> weighted mean -> ... ----
  auto freq_conv2 = stream.alloc_mdcontainer_async<complex_type>(n_freq, freq_total);
  auto padded_conv2 = stream.alloc_mdcontainer_async<float>(n_freq, padded_total);
  auto conv2_cropped = stream.alloc_mdcontainer_async<float>(n_freq, psf_nrow, psf_ncol);
  auto conv2_psf = stream.alloc_mdcontainer_async<float>(psf_nrow, psf_ncol);
  auto fwhm_mask = stream.alloc_mdcontainer_async<bool>(psf_nrow, psf_ncol);
  auto dilation_out = stream.alloc_mdcontainer_async<bool>(dirty_nrow, dirty_ncol);

  const float norm = 1.0f / static_cast<float>(padded_total);

  for (int i = 0; i < n_scales; i++) {
    // 5a. Batched multiply by G^2 in freq domain (with norm)
    kernel::multiply_psf_gauss_sq_batched_kernel<<<CEIL_DIV(freq_total, 256), 256, 0, cuda_stream>>>(
        freq_psf.data_handle(), gauss_kernels.data_handle() + static_cast<std::int64_t>(i) * freq_total,
        freq_conv2.data_handle(), freq_total, n_freq, norm);

    // 5b. Batched C2R back to space
    CUFFT_CALL(cufftExecC2R(ctx.plans_backward[0], freq_conv2.data_handle(), padded_conv2.data_handle()));

    // 5c. Batched fftshift + crop to PSF size
    linalg::fftshift_crop(padded_conv2.data_handle(), conv2_cropped.data_handle(), psf_nrow, psf_ncol, ctx.padded_nrow,
                          ctx.padded_ncol, ctx.padding_nrow, ctx.padding_ncol, n_freq, cuda_stream);

    // 5d. Weighted mean across channels -> 2D conv2_psf
    linalg::weighted_sum_async(stream, conv2_cropped, weights_freq, conv2_psf);

    // 5e. Max-reduce conv2_psf (synchronizes the stream)
    auto max_iter =
        thrust::max_element(thrust::cuda::par.on(cuda_stream), thrust::device_pointer_cast(conv2_psf.data_handle()),
                            thrust::device_pointer_cast(conv2_psf.data_handle() + psf_npix));
    float psf_max = 0.0f;
    CHECK_CUDA(cudaMemcpyAsync(&psf_max, thrust::raw_pointer_cast(max_iter), sizeof(float), cudaMemcpyDeviceToHost,
                               cuda_stream));
    stream.sync();

    // 5f. FWHM threshold: bool out = (conv2_psf > psf_max / 2)
    kernel::threshold_fwhm_kernel<<<CEIL_DIV(psf_npix, 256), 256, 0, cuda_stream>>>(
        conv2_psf.data_handle(), psf_max * 0.5f, fwhm_mask.data_handle(), psf_npix);

    // 5g. Bounding box of FWHM (synchronous host-side reduction)
    morphology::roi structure_roi = morphology::compute_mask_roi(stream, fwhm_mask);

    // Convert inclusive bounds (compute_mask_roi) to exclusive (binary_dilation expects exclusive max).
    structure_roi.xmax += 1;
    structure_roi.ymax += 1;

    // 5h. Dilate mask_per_scale[i] using FWHM as structuring element
    // binary_dilation only writes out[tid]=true on matches; zero the buffer first.
    CHECK_CUDA(cudaMemsetAsync(dilation_out.data_handle(), 0, dirty_npix * sizeof(bool), cuda_stream));
    core::device_span2d<bool> current_mask = emu::submdspan(mask_per_scale, i);
    morphology::binary_dilation(stream, current_mask, fwhm_mask, structure_roi, dilation_out);

    // 5i. Copy dilation result back into mask_per_scale[i]
    CHECK_CUDA(cudaMemcpyAsync(current_mask.data_handle(), dilation_out.data_handle(), dirty_npix * sizeof(bool),
                               cudaMemcpyDeviceToDevice, cuda_stream));
  }

  // ---- 6. Negate to "true=masked" convention and OR external_mask into every scale slice ----
  kernel::finalize_mask_kernel<<<CEIL_DIV(dirty_npix, 256), 256, 0, cuda_stream>>>(
      mask_per_scale.data_handle(), external_mask.data_handle(), dirty_npix, n_scales);
}
}  // namespace fast_deconv::common
