#pragma once
#include <cuda/std/mdspan>

#include <fast_deconv/algorithm/detail/rect.hpp>
#include <fast_deconv/algorithm/detail/spectral_fit.cuh>
#include <fast_deconv/algorithm/detail/update_mask.cuh>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/matrix/argmax.hpp>

#include <cstdio>
#include <cstring>
#include <vector>

namespace fast_deconv::algorithm::wscms::detail {

// Broadcast 2D mask (h, w) to 4D (nch, npol, h, w) via modulo
__global__ void broadcast_mask_2d_to_4d(const bool* __restrict__ mask_2d,
                                         bool* __restrict__ mask_4d,
                                         size_t spatial_size,
                                         size_t total_size)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    mask_4d[idx] = mask_2d[idx % spatial_size];
}

// scaled_dirty[i] -= psf_2[i] * gain_scaled * mask[i]
// mask is 2D (h, w) broadcast over channels/pols via modulo on spatial index
__global__ void subtract_scaled_dirty_kernel(core::device_span4d_fs psf_2,
                                              core::device_span4d_fs scaled_dirty,
                                              const bool* __restrict__ mask_2d,
                                              float gain_scaled,
                                              size_t spatial_size,
                                              uint n_elements)
{
    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) return;

    using index_t = typename core::device_span4d_fs::index_type;
    auto lin = tid;
    index_t i3 = lin % scaled_dirty.extent(3); lin /= scaled_dirty.extent(3);
    index_t i2 = lin % scaled_dirty.extent(2); lin /= scaled_dirty.extent(2);
    index_t i1 = lin % scaled_dirty.extent(1); lin /= scaled_dirty.extent(1);
    index_t i0 = lin;

    size_t sp_idx = static_cast<size_t>(i2) * scaled_dirty.extent(3) + i3;
    float m = mask_2d[sp_idx % spatial_size] ? 1.0f : 0.0f;

    scaled_dirty(i0, i1, i2, i3) -= psf_2(i0, i1, i2, i3) * gain_scaled * m;
}

// Subtract kernel with explicit element count for strided mdspans.
// out[i] = dirty[i] - psf[i] * coeffs[ch] * gain
__global__ void subtract_dirty_patch_kernel(core::device_span4d_fs psf,
                                             core::device_span4d_fs dirty,
                                             core::device_vect_f coeffs,
                                             core::device_span4d_fs out,
                                             float gain,
                                             uint n_elements)
{
    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) return;

    using index_t = typename core::device_span4d_fs::index_type;
    auto lin = tid;
    index_t i3 = lin % out.extent(3); lin /= out.extent(3);
    index_t i2 = lin % out.extent(2); lin /= out.extent(2);
    index_t i1 = lin % out.extent(1); lin /= out.extent(1);
    index_t i0 = lin;

    out(i0, i1, i2, i3) = dirty(i0, i1, i2, i3) - psf(i0, i1, i2, i3) * coeffs(i0) * gain;
}

// Extract 4D subview from 6D psfs at [facet_idx, scale_idx, :, :, :, :]
inline core::device_span4d_fs extract_psf_4d(core::span_6d<float>& psfs,
                                              int facet_idx, int scale_idx)
{
    const size_t nch  = psfs.extent(2);
    const size_t npol = psfs.extent(3);
    const size_t h    = psfs.extent(4);
    const size_t w    = psfs.extent(5);

    const size_t offset = static_cast<size_t>(facet_idx) * psfs.extent(1) * nch * npol * h * w
                        + static_cast<size_t>(scale_idx) * nch * npol * h * w;

    float* ptr = psfs.data_handle() + offset;
    std::array<size_t, 4> strides = {npol * h * w, h * w, w, 1};

    return core::device_span4d_fs(ptr, emu::layout_stride::mapping<core::dims<4>>(
        core::dims<4>(nch, npol, h, w), strides));
}

// Create 4D strided subview at patch region from contiguous base.
// Rect.x maps to the row dimension (h), Rect.y to the column dimension (w).
// So patch rows = rect.width() and patch cols = rect.height().
inline core::device_span4d_fs make_patch_subview(float* base_ptr,
                                                   size_t nch, size_t npol,
                                                   size_t full_h, size_t full_w,
                                                   const Rect& rect)
{
    float* ptr = base_ptr + rect.x0 * full_w + rect.y0;
    std::array<size_t, 4> strides = {npol * full_h * full_w, full_h * full_w, full_w, 1};
    return core::device_span4d_fs(ptr, emu::layout_stride::mapping<core::dims<4>>(
        core::dims<4>(nch, npol, static_cast<size_t>(rect.width()),
                      static_cast<size_t>(rect.height())), strides));
}

// Create 4D strided subview at patch region from existing strided 4D.
// Rect.x maps to rows (dim 2), Rect.y maps to cols (dim 3).
inline core::device_span4d_fs make_patch_subview_from_4d(core::device_span4d_fs& src,
                                                          const Rect& rect)
{
    const size_t nch  = src.extent(0);
    const size_t npol = src.extent(1);
    const size_t stride_ch  = src.mapping().stride(0);
    const size_t stride_pol = src.mapping().stride(1);
    const size_t stride_row = src.mapping().stride(2);

    float* ptr = src.data_handle() + rect.x0 * stride_row + rect.y0;
    std::array<size_t, 4> strides = {stride_ch, stride_pol, stride_row, 1};
    return core::device_span4d_fs(ptr, emu::layout_stride::mapping<core::dims<4>>(
        core::dims<4>(nch, npol, static_cast<size_t>(rect.width()),
                      static_cast<size_t>(rect.height())), strides));
}


void wscms_minor_cycle(core::span_4d<float> dirty,
                       core::span_4d<float> scaled_dirty,
                       std::uint32_t scale_idx,
                       const MinorCycleContext& ctx,
                       ComponentBuffer& components,
                       core::stream_resources& resources)
{
    auto stream = resources.stream;

    const size_t nch  = dirty.extent(0);
    const size_t npol = dirty.extent(1);
    const size_t h    = dirty.extent(2);
    const size_t w    = dirty.extent(3);
    const size_t spatial_size = h * w;
    const size_t total_4d_size = nch * npol * h * w;
    const int order = static_cast<int>(ctx.Xdes.extent(1));
    const size_t psf_h = ctx.psfs.extent(4);
    const size_t psf_w = ctx.psfs.extent(5);

    // Allocate 4D bool mask for argmax (broadcast from 2D mask)
    bool* d_mask_4d;
    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_mask_4d),
                                sizeof(bool) * total_4d_size, stream));

    // Allocate spectral fitting workspace
    auto fit_ws = SpectralFitWorkspace::allocate(static_cast<int>(nch), order, stream);

    // Broadcast mask 2D -> 4D bool
    broadcast_mask_2d_to_4d<<<CEIL_DIV(total_4d_size, 256), 256, 0, stream>>>(
        ctx.mask.data_handle(), d_mask_4d, spatial_size, total_4d_size);
    CHECK_LAST_CUDA_ERROR();

    // Initial peak finding
    auto [peak_idx, peak_val] = matrix::argmax(
        scaled_dirty.data_handle(), d_mask_4d, total_4d_size, ctx.do_abs, resources);

    // Unravel: layout_right (nch, npol, h, w) -> spatial coords
    size_t peak_x = (peak_idx / w) % h;
    size_t peak_y = peak_idx % w;

    float threshold = ctx.peak_factor * peak_val;

    // Update mask: mask &= (|scaled_dirty| > threshold)
    update_mask(ctx.mask, scaled_dirty, threshold, ctx.do_abs, resources);

    // Re-broadcast updated mask to 4D bool and compute 2D float mask
    broadcast_mask_2d_to_4d<<<CEIL_DIV(total_4d_size, 256), 256, 0, stream>>>(
        ctx.mask.data_handle(), d_mask_4d, spatial_size, total_4d_size);
    CHECK_LAST_CUDA_ERROR();

    int sub_iter = 0;
    auto psfs_mut = ctx.psfs;
    auto psfs_2_mut = ctx.psfs_2;

    while (peak_val > threshold && sub_iter < ctx.n_subminor_iter) {
        int ix = static_cast<int>(peak_x);
        int iy = static_cast<int>(peak_y);

        // Get facet index
        int facet_idx;
        CHECK_CUDA(cudaMemcpyAsync(&facet_idx,
            ctx.map_pixels_facets.data_handle() + ix * static_cast<int>(w) + iy,
            sizeof(int), cudaMemcpyDeviceToHost, stream));
        resources.sync();

        // Get gain
        float gain = 0.0f;
        CHECK_CUDA(cudaMemcpy(&gain,
            ctx.gains.data_handle() + facet_idx * ctx.gains.extent(1) + scale_idx,
            sizeof(float), cudaMemcpyDeviceToHost));

        // Get scaled_dirty at peak for scaled subtraction
        float sd_peak_val = 0.0f;
        CHECK_CUDA(cudaMemcpy(&sd_peak_val,
            scaled_dirty.data_handle() + peak_x * w + peak_y,
            sizeof(float), cudaMemcpyDeviceToHost));

        // Spectral fitting
        float h_coeffs[MAX_SPECTRAL_ORDER] = {};
        int n_coeffs = 0;
        spectral_fit(ctx, dirty, ix, iy, fit_ws, h_coeffs, n_coeffs, resources);

        // Save component
        ComponentEntry& entry = components.entries[sub_iter];
        entry.x = ix;
        entry.y = iy;
        entry.scale_idx = static_cast<int>(scale_idx);
        entry.gain = gain;
        entry.n_coeffs = n_coeffs;
        std::memcpy(entry.coeffs, h_coeffs, sizeof(float) * n_coeffs);

        // Compute aligned patch edges
        auto [img_rect, psf_rect] = compute_aligned_patch_edges(
            ix, iy,
            static_cast<int>(h), static_cast<int>(w),
            static_cast<int>(psf_h), static_cast<int>(psf_w));

        if (!img_rect.empty()) {
            auto psf_4d = extract_psf_4d(psfs_mut, facet_idx, static_cast<int>(scale_idx));
            auto psf_2_4d = extract_psf_4d(psfs_2_mut, facet_idx, static_cast<int>(scale_idx));

            auto dirty_patch = make_patch_subview(dirty.data_handle(), nch, npol, h, w, img_rect);
            auto scaled_dirty_patch = make_patch_subview(scaled_dirty.data_handle(), nch, npol, h, w, img_rect);
            auto psf_patch = make_patch_subview_from_4d(psf_4d, psf_rect);
            auto psf_2_patch = make_patch_subview_from_4d(psf_2_4d, psf_rect);

            // per_channel coefficients from spectral fit
            core::device_vect_f coeffs_span(fit_ws.d_per_channel_f, nch);

            // Compute logical element count (product of extents)
            // Cannot rely on .size() for layout_stride as it may return required_span_size
            // rect.width() = row extent, rect.height() = col extent
            const size_t patch_elements = nch * npol *
                static_cast<size_t>(img_rect.width()) * static_cast<size_t>(img_rect.height());

            // 1. dirty -= psf * per_channel_coeffs * gain
            subtract_dirty_patch_kernel
                <<<CEIL_DIV(patch_elements, size_t{256}), 256, 0, stream>>>(
                    psf_patch, dirty_patch, coeffs_span, dirty_patch, gain,
                    static_cast<uint>(patch_elements));
            CHECK_LAST_CUDA_ERROR();

            // 2. scaled_dirty -= psf_2 * (sd_peak_val * gain) * mask
            float gain_scaled = sd_peak_val * gain;
            subtract_scaled_dirty_kernel
                <<<CEIL_DIV(patch_elements, size_t{256}), 256, 0, stream>>>(
                    psf_2_patch, scaled_dirty_patch,
                    ctx.mask.data_handle(), gain_scaled, spatial_size,
                    static_cast<uint>(patch_elements));
            CHECK_LAST_CUDA_ERROR();
        }

        // Next peak
        auto [next_peak_idx, next_peak_val] = matrix::argmax(
            scaled_dirty.data_handle(), d_mask_4d, total_4d_size, ctx.do_abs, resources);

        peak_val = next_peak_val;
        peak_x = (next_peak_idx / w) % h;
        peak_y = next_peak_idx % w;

        sub_iter++;
    }

    components.count = sub_iter;

    // Cleanup
    fit_ws.free(stream);
    CHECK_CUDA(cudaFreeAsync(d_mask_4d, stream));
}

}  // namespace fast_deconv::algorithm::wscms::detail
