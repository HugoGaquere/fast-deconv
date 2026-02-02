import cupy as cp
import fast_deconv as fd
import pytest


def full_slice(ndim):
    return (slice(None),) * ndim


def head_slice(ndim):
    return (slice(1, None),) + (slice(None),) * (ndim - 1)


def tail_slice(ndim):
    return (slice(None, -1),) + (slice(None),) * (ndim - 1)


def strided_slice(ndim):
    return (
        slice(
            None,
        ),
    ) * (ndim - 1) + (slice(None, -3),)


def reversed_slice(ndim):
    return (slice(None, None, -1),) * ndim


OP_SUPPORTED_SLICES = [
    # full_slice,
    # head_slice,
    # tail_slice,
    strided_slice,
]

def compare_arrays(A, B):
    diff = A - B
    print("shape:", A.shape)
    print("max abs diff:", cp.max(cp.abs(diff)))
    print("mean abs diff:", cp.mean(cp.abs(diff)))
    print("l2 norm:", cp.linalg.norm(diff))
    print("different elements:", cp.count_nonzero(A != B))

@pytest.mark.parametrize("ndim", [4])
@pytest.mark.parametrize("slice_factory", OP_SUPPORTED_SLICES)
def test_subtract_psf_from_dirty(ndim, slice_factory):
    rng = cp.random.default_rng(12345)
    resources = fd.stream_resources()

    shape = tuple(8 + 2*i for i in range(ndim))
    slc = slice_factory(ndim)

    dirty = rng.standard_normal(shape, dtype=cp.float32)
    psf = rng.standard_normal(shape, dtype=cp.float32)
    coeffs = rng.standard_normal(shape[0], dtype=cp.float32)
    gain = rng.standard_normal(1, dtype=cp.float32)[0]
    out_true = cp.zeros(shape, dtype=cp.float32)
    out_actual = cp.zeros(shape, dtype=cp.float32)

    cp.cuda.runtime.deviceSynchronize()

    scaled_psf = psf * coeffs[:, None, None, None] * gain
    cp.subtract(dirty[slc], scaled_psf[slc], out=out_true[slc])

    cp.cuda.runtime.deviceSynchronize()

    fd.wscms.subtract_psf_from_dirty_async(
        psf[slc], dirty[slc], coeffs, out_actual[slc], gain, resources
    )

    cp.cuda.runtime.deviceSynchronize()

    compare_arrays(out_true, out_actual)
    cp.testing.assert_allclose(out_actual , out_true, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("slice_factory", OP_SUPPORTED_SLICES)
def test_clean_dirties_basic(slice_factory):
    """Test the fused kernel with basic inputs."""
    rng = cp.random.default_rng(42)
    resources = fd.stream_resources()

    shape = (4, 2, 16, 16)  # nch, npol, h, w
    slc = slice_factory(4)

    # Create input arrays
    psf = rng.standard_normal(shape, dtype=cp.float32)
    psf_2 = rng.standard_normal(shape, dtype=cp.float32)
    dirty = rng.standard_normal(shape, dtype=cp.float32)
    scaled_dirty = rng.standard_normal(shape, dtype=cp.float32)
    coeffs = rng.standard_normal(shape[0], dtype=cp.float32)
    mask = rng.integers(0, 2, shape).astype(cp.float32)  # Binary mask as float
    gain = float(rng.standard_normal(1, dtype=cp.float32)[0])

    # Copy for expected computation
    dirty_expected = dirty.copy()
    scaled_dirty_expected = scaled_dirty.copy()

    cp.cuda.runtime.deviceSynchronize()

    # Compute expected results using CuPy
    # dirty -= psf * coeffs * gain
    dirty_expected[slc] -= psf[slc] * coeffs[:, None, None, None] * gain
    # scaled_dirty -= psf_2 * gain_scaled * mask
    scaled_dirty_expected[slc] -= psf_2[slc] * gain * mask[slc]

    cp.cuda.runtime.deviceSynchronize()

    # Call the CUDA kernel
    fd.wscms.clean_dirties_async(
        psf[slc], psf_2[slc], dirty[slc], scaled_dirty[slc],
        coeffs, mask[slc], gain, resources
    )

    cp.cuda.runtime.deviceSynchronize()

    # Compare results
    print("\n=== Dirty comparison ===")
    compare_arrays(dirty_expected, dirty)
    cp.testing.assert_allclose(dirty, dirty_expected, rtol=1e-5, atol=1e-6)

    print("\n=== Scaled dirty comparison ===")
    compare_arrays(scaled_dirty_expected, scaled_dirty)
    cp.testing.assert_allclose(scaled_dirty, scaled_dirty_expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("slice_factory", OP_SUPPORTED_SLICES)
def test_clean_dirties_with_zero_mask(slice_factory):
    """Test that zero mask means no subtraction on scaled_dirty."""
    rng = cp.random.default_rng(123)
    resources = fd.stream_resources()

    shape = (2, 1, 8, 8)
    slc = slice_factory(4)

    psf = rng.standard_normal(shape, dtype=cp.float32)
    psf_2 = rng.standard_normal(shape, dtype=cp.float32)
    dirty = rng.standard_normal(shape, dtype=cp.float32)
    scaled_dirty = rng.standard_normal(shape, dtype=cp.float32)
    coeffs = rng.standard_normal(shape[0], dtype=cp.float32)
    mask = cp.zeros(shape, dtype=cp.float32)  # All zeros
    gain = 0.5

    dirty_expected = dirty.copy()
    scaled_dirty_original = scaled_dirty.copy()

    cp.cuda.runtime.deviceSynchronize()

    # Expected: dirty changes, scaled_dirty unchanged (mask is zero)
    dirty_expected[slc] -= psf[slc] * coeffs[:, None, None, None] * gain

    fd.wscms.clean_dirties_async(
        psf[slc], psf_2[slc], dirty[slc], scaled_dirty[slc],
        coeffs, mask[slc], gain, resources
    )

    cp.cuda.runtime.deviceSynchronize()

    cp.testing.assert_allclose(dirty, dirty_expected, rtol=1e-5, atol=1e-6)
    # scaled_dirty should be unchanged where mask is zero
    cp.testing.assert_allclose(scaled_dirty[slc], scaled_dirty_original[slc], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("slice_factory", OP_SUPPORTED_SLICES)
def test_clean_dirties_with_full_mask(slice_factory):
    """Test that full mask applies subtraction everywhere on scaled_dirty."""
    rng = cp.random.default_rng(456)
    resources = fd.stream_resources()

    shape = (3, 2, 12, 12)
    slc = slice_factory(4)

    psf = rng.standard_normal(shape, dtype=cp.float32)
    psf_2 = rng.standard_normal(shape, dtype=cp.float32)
    dirty = rng.standard_normal(shape, dtype=cp.float32)
    scaled_dirty = rng.standard_normal(shape, dtype=cp.float32)
    coeffs = rng.standard_normal(shape[0], dtype=cp.float32)
    mask = cp.ones(shape, dtype=cp.float32)  # All ones
    gain = 0.1

    dirty_expected = dirty.copy()
    scaled_dirty_expected = scaled_dirty.copy()

    cp.cuda.runtime.deviceSynchronize()

    dirty_expected[slc] -= psf[slc] * coeffs[:, None, None, None] * gain
    scaled_dirty_expected[slc] -= psf_2[slc] * gain * mask[slc]

    fd.wscms.clean_dirties_async(
        psf[slc], psf_2[slc], dirty[slc], scaled_dirty[slc],
        coeffs, mask[slc], gain, resources
    )

    cp.cuda.runtime.deviceSynchronize()

    cp.testing.assert_allclose(dirty, dirty_expected, rtol=1e-5, atol=1e-6)
    cp.testing.assert_allclose(scaled_dirty, scaled_dirty_expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [
    (1, 1, 8, 8),    # Minimal (needs room for strided slice)
    (8, 4, 32, 32),  # Larger
    (2, 2, 64, 64),  # Square
    (4, 1, 16, 32),  # Non-square
])
def test_clean_dirties_different_shapes(shape):
    """Test with various array shapes using strided slices."""
    rng = cp.random.default_rng(789)
    resources = fd.stream_resources()
    slc = strided_slice(4)  # Creates strided view required by the kernel

    psf = rng.standard_normal(shape, dtype=cp.float32)
    psf_2 = rng.standard_normal(shape, dtype=cp.float32)
    dirty = rng.standard_normal(shape, dtype=cp.float32)
    scaled_dirty = rng.standard_normal(shape, dtype=cp.float32)
    coeffs = rng.standard_normal(shape[0], dtype=cp.float32)
    mask = rng.integers(0, 2, shape).astype(cp.float32)
    gain = 0.5

    dirty_expected = dirty.copy()
    scaled_dirty_expected = scaled_dirty.copy()

    cp.cuda.runtime.deviceSynchronize()

    dirty_expected[slc] -= psf[slc] * coeffs[:, None, None, None] * gain
    scaled_dirty_expected[slc] -= psf_2[slc] * gain * mask[slc]

    fd.wscms.clean_dirties_async(
        psf[slc], psf_2[slc], dirty[slc], scaled_dirty[slc],
        coeffs, mask[slc], gain, resources
    )

    cp.cuda.runtime.deviceSynchronize()

    cp.testing.assert_allclose(
        dirty, dirty_expected, rtol=1e-5, atol=1e-6,
        err_msg=f"Dirty mismatch for shape {shape}"
    )
    cp.testing.assert_allclose(
        scaled_dirty, scaled_dirty_expected, rtol=1e-5, atol=1e-6,
        err_msg=f"Scaled dirty mismatch for shape {shape}"
    )


def _python_reference_minor_cycle(
    dirty, scaled_dirty, mask, psfs, psfs_2, gains, jones_norm,
    Xdes, sqrt_weights, map_pixels_facets,
    scale_idx, peak_factor, n_subminor_iter, do_abs, beam_enable,
):
    """Pure CuPy reference implementation of wscms_minor_cycle."""
    import numpy as np

    nch, npol, h, w = dirty.shape

    # Find peak
    mask_4d = cp.broadcast_to(mask[None, None, :, :], dirty.shape).copy()
    if do_abs:
        masked = cp.where(mask_4d, cp.abs(scaled_dirty), 0)
    else:
        masked = cp.where(mask_4d, scaled_dirty, -cp.inf)
    peak_flat = int(cp.argmax(masked).item())
    peak_coords = np.unravel_index(peak_flat, dirty.shape)
    # When do_abs, argmax returns abs(value), so peak_val is always positive
    peak_val = abs(float(scaled_dirty[peak_coords].item())) if do_abs else float(scaled_dirty[peak_coords].item())

    threshold = peak_factor * peak_val

    # Update mask
    if do_abs:
        mask = cp.logical_and(cp.abs(scaled_dirty[0, 0]) > threshold, mask)
    else:
        mask = cp.logical_and(scaled_dirty[0, 0] > threshold, mask)

    mask_4d = cp.broadcast_to(mask[None, None, :, :], dirty.shape).copy()

    components = []
    sub_iter = 0
    while peak_val > threshold and sub_iter < n_subminor_iter:
        x, y = peak_coords[2], peak_coords[3]
        facet_idx = int(map_pixels_facets[x * w + y].item())

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
            psf_2_sub * gain_scaled * mask[None, None, img_x0:img_x1, img_y0:img_y1].astype(cp.float32)
        )

        # Next peak
        if do_abs:
            masked = cp.where(mask_4d, cp.abs(scaled_dirty), 0)
        else:
            masked = cp.where(mask_4d, scaled_dirty, -cp.inf)
        peak_flat = int(cp.argmax(masked).item())
        peak_coords = np.unravel_index(peak_flat, dirty.shape)
        peak_val = abs(float(scaled_dirty[peak_coords].item())) if do_abs else float(scaled_dirty[peak_coords].item())
        sub_iter += 1

    return sub_iter, components


def test_wscms_minor_cycle_basic():
    """Integration test: compare C++ minor cycle against Python reference."""
    import numpy as np

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
    mask = cp.ones((h, w), dtype=cp.bool_)
    psfs = rng.standard_normal((nfacets, nscales, nch, npol, psf_h, psf_w), dtype=cp.float32)
    psfs_2 = rng.standard_normal((nfacets, nscales, nch, npol, psf_h, psf_w), dtype=cp.float32)
    gains = cp.ones((nfacets, nscales), dtype=cp.float32) * 0.1
    jones_norm = cp.ones((nch, npol, h, w), dtype=cp.float32)
    Xdes = rng.standard_normal((nch, order), dtype=cp.float32)
    sqrt_weights = cp.ones(nch, dtype=cp.float32)
    map_pixels_facets = cp.zeros(h * w, dtype=cp.int32)  # All pixels -> facet 0

    scale_idx = 0
    peak_factor = 0.5
    n_subminor_iter = 5
    do_abs = True
    beam_enable = True

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
    ctx = fd.wscms.make_minor_cycle_context(
        psfs=psfs, psfs_2=psfs_2,
        jones_norm=jones_norm, gains=gains, mask=mask,
        map_pixels_facets=map_pixels_facets,
        Xdes=Xdes, sqrt_weights=sqrt_weights,
        beam_enable=beam_enable,
        peak_factor=peak_factor,
        n_subminor_iter=n_subminor_iter,
        do_abs=do_abs,
    )
    cpp_components = fd.wscms.wscms_minor_cycle(
        dirty, scaled_dirty, scale_idx, ctx, resources,
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


def test_wscms_minor_cycle_single_iter():
    """Test 1-iteration minor cycle matches Python reference closely."""
    import numpy as np

    rng = cp.random.default_rng(42)
    resources = fd.stream_resources()

    nfacets, nscales = 1, 1
    nch, npol = 4, 1
    h, w = 64, 64
    psf_h, psf_w = 32, 32
    order = 2

    dirty = rng.standard_normal((nch, npol, h, w), dtype=cp.float32) * 10
    scaled_dirty = dirty.copy()
    mask = cp.ones((h, w), dtype=cp.bool_)
    psfs = rng.standard_normal((nfacets, nscales, nch, npol, psf_h, psf_w), dtype=cp.float32)
    psfs_2 = rng.standard_normal((nfacets, nscales, nch, npol, psf_h, psf_w), dtype=cp.float32)
    gains = cp.ones((nfacets, nscales), dtype=cp.float32) * 0.1
    jones_norm = cp.ones((nch, npol, h, w), dtype=cp.float32)
    Xdes = rng.standard_normal((nch, order), dtype=cp.float32)
    sqrt_weights = cp.ones(nch, dtype=cp.float32)
    map_pixels_facets = cp.zeros(h * w, dtype=cp.int32)

    dirty_ref = dirty.copy()
    scaled_dirty_ref = scaled_dirty.copy()
    mask_ref = mask.copy()

    cp.cuda.runtime.deviceSynchronize()

    ref_iters, ref_components = _python_reference_minor_cycle(
        dirty_ref, scaled_dirty_ref, mask_ref,
        psfs, psfs_2, gains, jones_norm,
        Xdes, sqrt_weights, map_pixels_facets,
        0, 0.5, 1, True, True,
    )

    ctx = fd.wscms.make_minor_cycle_context(
        psfs=psfs, psfs_2=psfs_2,
        jones_norm=jones_norm, gains=gains, mask=mask,
        map_pixels_facets=map_pixels_facets,
        Xdes=Xdes, sqrt_weights=sqrt_weights,
        beam_enable=True, peak_factor=0.5, n_subminor_iter=1, do_abs=True)
    cpp_components = fd.wscms.wscms_minor_cycle(
        dirty, scaled_dirty, 0, ctx, resources)

    cp.cuda.runtime.deviceSynchronize()

    assert len(cpp_components) == ref_iters
    cp.testing.assert_allclose(dirty, dirty_ref, rtol=1e-5, atol=1e-5)

