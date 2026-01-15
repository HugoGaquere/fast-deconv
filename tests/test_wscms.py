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

    shape = tuple(8 + i for i in range(ndim))
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

    fd.wscms.subtract_psf_from_dirty(
        psf[slc], dirty[slc], coeffs, out_actual[slc], gain, resources
    )

    cp.cuda.runtime.deviceSynchronize()

    compare_arrays(out_true, out_actual)
    cp.testing.assert_allclose(out_actual , out_true, rtol=1e-6, atol=1e-7)

