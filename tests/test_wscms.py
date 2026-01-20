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

