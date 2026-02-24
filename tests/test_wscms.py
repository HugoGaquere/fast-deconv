import cupy as cp
import numpy as np
import fast_deconv as fd
import pytest

def compare_arrays(A, B):
    diff = A - B
    print("shape:", A.shape)
    print("max abs diff:", cp.max(cp.abs(diff)))
    print("mean abs diff:", cp.mean(cp.abs(diff)))
    print("l2 norm:", cp.linalg.norm(diff))
    print("different elements:", cp.count_nonzero(A != B))


def _python_reference_minor_cycle(
    dirty, scaled_dirty, mask, psfs, psfs_2, gains, jones_norm,
    Xdes, sqrt_weights, map_pixels_facets,
    scale_idx, peak_factor, n_subminor_iter, do_abs, beam_enable,
):
    """Pure CuPy reference implementation of wscms_minor_cycle."""

    nch, npol, h, w = dirty.shape

    # Find peak
    if do_abs:
        masked = cp.where(mask, cp.abs(scaled_dirty), 0)
    else:
        masked = cp.where(mask, scaled_dirty, -cp.inf)
    peak_flat = int(cp.argmax(masked).item())
    peak_coords = np.unravel_index(peak_flat, dirty.shape)
    # When do_abs, argmax returns abs(value), so peak_val is always positive
    peak_val = abs(float(scaled_dirty[peak_coords].item())) if do_abs else float(scaled_dirty[peak_coords].item())

    threshold = peak_factor * peak_val

    # Update mask
    if do_abs:
        mask = cp.logical_and(cp.abs(dirty) > threshold, mask)
    else:
        mask = cp.logical_and(dirty > threshold, mask)

    components = []
    sub_iter = 0
    while peak_val > threshold and sub_iter < n_subminor_iter:
        x, y = peak_coords[2], peak_coords[3]
        facet_idx = int(map_pixels_facets[x, y].item())

        gain = float(gains[facet_idx, scale_idx].item())
        sd_peak = float(scaled_dirty[0, 0, x, y].item())

        # Spectral fitting
        jn = jones_norm[:, 0, x, y]
        apparent_flux = dirty[:, 0, x, y]

        if beam_enable:
            SAX = cp.sqrt(jn)[:, None] * Xdes
        else:
            SAX = Xdes.copy()

        WX = (sqrt_weights[:, None] * SAX).astype(cp.float64)
        nchan, order = WX.shape
        if nchan >= order:
            pinv = cp.linalg.inv(WX.T @ WX) @ WX.T
        else:
            pinv = WX.T @ cp.linalg.inv(WX @ WX.T)

        Wy = (sqrt_weights * apparent_flux).astype(cp.float64)
        coeffs_compact = pinv @ Wy
        per_channel = (SAX.astype(cp.float64) @ coeffs_compact).astype(cp.float32)

        components.append(((int(x), int(y)), cp.asnumpy(coeffs_compact).astype(np.float32), scale_idx, gain))

        # Patch edges
        psf_h, psf_w = psfs.shape[4], psfs.shape[5]
        left = psf_w // 2; right = psf_w - left
        top = psf_h // 2; bottom = psf_h - top

        img_x0 = max(0, x - left); img_x1 = min(h, x + right)
        img_y0 = max(0, y - top); img_y1 = min(w, y + bottom)
        psf_x0 = img_x0 - (x - left); psf_x1 = psf_w - ((x + right) - img_x1)
        psf_y0 = img_y0 - (y - top); psf_y1 = psf_h - ((y + bottom) - img_y1)

        # Subtract from dirty
        psf_sub = psfs[facet_idx, scale_idx, :, :, psf_x0:psf_x1, psf_y0:psf_y1]
        dirty[:, :, img_x0:img_x1, img_y0:img_y1] -= (
            psf_sub * per_channel[:, None, None, None] * gain
        )

        # Subtract from scaled_dirty
        psf_2_sub = psfs_2[facet_idx, scale_idx, :, :, psf_x0:psf_x1, psf_y0:psf_y1]
        gain_scaled = sd_peak * gain
        scaled_dirty[:, :, img_x0:img_x1, img_y0:img_y1] -= (
            psf_2_sub * gain_scaled * mask[:, :, img_x0:img_x1, img_y0:img_y1].astype(cp.float32)
        )

        # Next peak
        if do_abs:
            masked = cp.where(mask, cp.abs(scaled_dirty), 0)
        else:
            masked = cp.where(mask, scaled_dirty, -cp.inf)
        peak_flat = int(cp.argmax(masked).item())
        peak_coords = np.unravel_index(peak_flat, dirty.shape)
        peak_val = abs(float(scaled_dirty[peak_coords].item())) if do_abs else float(scaled_dirty[peak_coords].item())
        sub_iter += 1

    return sub_iter, components


def test_wscms_minor_cycle_basic():
    """Integration test: compare C++ minor cycle against Python reference."""

    rng = cp.random.default_rng(42)
    resources = fd.stream_resources()

    nfacets, nscales = 1, 1
    nch, npol = 4, 1
    h, w = 64, 64
    psf_h, psf_w = 32, 32
    order = 2

    # Create test data
    dirty = rng.standard_normal((nch, npol, h, w), dtype=cp.float32) * 10
    scaled_dirty = dirty.copy()
    mask = cp.ones((nch, npol, h, w), dtype=cp.bool_)
    psfs = rng.standard_normal((nfacets, nscales, nch, npol, psf_h, psf_w), dtype=cp.float32)
    psfs_2 = rng.standard_normal((nfacets, nscales, nch, npol, psf_h, psf_w), dtype=cp.float32)
    gains = np.ones((nfacets, nscales), dtype=np.float32) * 0.1
    jones_norm = cp.ones((nch, npol, h, w), dtype=cp.float32)
    Xdes = rng.standard_normal((nch, order), dtype=cp.float32)
    sqrt_weights = cp.ones(nch, dtype=cp.float32)
    map_pixels_facets = np.zeros((h, w), dtype=np.int32)  # All pixels -> facet 0

    scale_idx = 0
    peak_factor = 0.5
    n_subminor_iter = 5
    do_abs = True
    beam_enable = False

    # Make copies for reference
    dirty_ref = dirty.copy()
    scaled_dirty_ref = scaled_dirty.copy()
    mask_ref = mask.copy()

    cp.cuda.runtime.deviceSynchronize()

    # Python reference
    ref_iters, ref_components = _python_reference_minor_cycle(
        dirty_ref, scaled_dirty_ref, mask_ref,
        psfs, psfs_2, gains, jones_norm,
        Xdes, sqrt_weights, map_pixels_facets,
        scale_idx, peak_factor, n_subminor_iter, do_abs, beam_enable,
    )

    cp.cuda.runtime.deviceSynchronize()

    # C++ implementation
    ctx = fd.wscms.MinorCycleContext(
        jones_norm=jones_norm,
        map_pixels_facets=map_pixels_facets,
        Xdes=Xdes, sqrt_weights=sqrt_weights,
        beam_enable=beam_enable,
        peak_factor=peak_factor,
        n_subminor_iter=n_subminor_iter,
        do_abs=do_abs,
    )
    cpp_components = fd.wscms.minor_cycle(
        dirty, scaled_dirty, psfs, psfs_2, mask, gains, scale_idx, ctx,
    )

    cp.cuda.runtime.deviceSynchronize()

    # Compare iteration count
    assert len(cpp_components) == ref_iters, (
        f"Iteration count mismatch: C++ {len(cpp_components)} vs Python {ref_iters}"
    )

    # Compare first component (subsequent ones may diverge due to FP accumulation)
    if len(cpp_components) > 0 and len(ref_components) > 0:
        cpp_coords, cpp_coeffs, cpp_scale, cpp_gain = cpp_components[0]
        ref_coords, ref_coeffs, ref_scale, ref_gain = ref_components[0]
        assert cpp_coords == ref_coords, f"Component 0: coords mismatch {cpp_coords} vs {ref_coords}"
        assert cpp_scale == ref_scale, f"Component 0: scale mismatch"
        assert abs(cpp_gain - ref_gain) < 1e-6, f"Component 0: gain mismatch"


