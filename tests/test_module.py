import cupy as cp
import numpy as np
import fast_deconv
import pytest


def full_slice(ndim):
    return (slice(None),) * ndim


def head_slice(ndim):
    return (slice(1, None),) + (slice(None),) * (ndim - 1)


def tail_slice(ndim):
    return (slice(None, -1),) + (slice(None),) * (ndim - 1)


def strided_slice(ndim):
    return (slice(None,),) * (ndim - 1) + (slice(None, -3),)


def reversed_slice(ndim):
    return (slice(None, None, -1),) * ndim


def argmax_cupy(data, mask, do_abs: bool):
    """
    Reference argmax using CuPy.
    """
    data = cp.asarray(data)
    mask = cp.asarray(mask, dtype=cp.bool_)
    metric = cp.abs(data) if do_abs else data
    masked = cp.where(mask, metric, -cp.inf)
    flat_idx = int(cp.argmax(masked).item())
    orig_val = data.ravel()[flat_idx]
    ret_val = float(cp.abs(orig_val).item()) if do_abs else float(orig_val.item())
    return flat_idx, ret_val


ARGMAX_SUPPORTED_SLICES = [
    full_slice,
    head_slice,
    tail_slice,
    # strided_slice,
]


@pytest.mark.parametrize("ndim", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("slice_factory", ARGMAX_SUPPORTED_SLICES)
@pytest.mark.parametrize("do_abs", [False, True])
def test_argmax(ndim, slice_factory, do_abs):
    rng = cp.random.default_rng(12345)
    resources = fast_deconv.stream_resources()

    shape = tuple(8 + i for i in range(ndim))

    A = rng.standard_normal(shape, dtype=cp.float32)
    mask = rng.random(shape) > 0.3
    slc = slice_factory(ndim)
    cp.cuda.runtime.deviceSynchronize()

    actual_idx, actual_val = fast_deconv.matrix.argmax(
        A[slc], mask[slc], do_abs, resources
    )
    true_idx, true_val = argmax_cupy(A[slc], mask[slc], do_abs)
    cp.cuda.runtime.deviceSynchronize()

    assert actual_idx == true_idx
    assert actual_val == true_val


SUBTRACT_SUPPORTED_SLICES = [
    full_slice,
    head_slice,
    tail_slice,
    strided_slice,
]

@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("slice_factory", SUBTRACT_SUPPORTED_SLICES)
def test_supported_subtract_rank_generic(ndim, slice_factory):
    rng = cp.random.default_rng(12345)
    resources = fast_deconv.stream_resources()

    shape = tuple(8 + i for i in range(ndim))

    A = rng.standard_normal(shape, dtype=cp.float32)
    B = rng.standard_normal(shape, dtype=cp.float32)
    C = cp.zeros_like(A, dtype=cp.float32)
    A_h = cp.asnumpy(A)
    B_h = cp.asnumpy(B)
    slc = slice_factory(ndim)
    cp.cuda.runtime.deviceSynchronize()

    fast_deconv.matrix.subtract(A[slc], B[slc], C[slc], resources)
    C_host = cp.asnumpy(C)

    np.testing.assert_array_equal(C_host[slc], A_h[slc] - B_h[slc])
