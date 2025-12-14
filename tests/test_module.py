import cupy as cp
import fast_deconv
import pytest


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


@pytest.mark.parametrize("shape", [(16, 16), (31, 17), (128, 64)])
@pytest.mark.parametrize("do_abs", [False, True])
def test_argmax(shape, do_abs):
    rng = cp.random.default_rng(12345)
    resources = fast_deconv.stream_resources()

    A = rng.standard_normal(shape, dtype=cp.float32)
    mask = rng.random(shape) > 0.3

    true_idx, true_val = argmax_cupy(A, mask, do_abs)
    actual_idx, actual_val = fast_deconv.matrix.argmax(A, mask, do_abs, resources)

    assert actual_idx == true_idx
    assert actual_val == true_val

    # TODO: When slice will be supported:
    # Different slicing patterns to generate non-trivial views
    # slice_cases = [
    #     full array
    #     (slice(None), slice(None)),
    #     (slice(0, 3), slice(0, 2)),
    #     strided rows & cols
    #     (slice(None, None, 2), slice(None, None, 3)),
    #     interior sub-block
    #     (slice(1, -1), slice(2, -2)),
    #     tail block
    #     (slice(shape[0] // 4, None), slice(None, shape[1] // 2)),
    # ]
    #
    # for s0, s1 in slice_cases:
    #     A_view = A[s0, s1]
    #     mask_view = mask[s0, s1]
    #
    #     # Skip degenerate empty views (can happen for small shapes)
    #     if A_view.size == 0:
    #         continue
    #
    #     true_idx, true_val = argmax_cupy(A_view, mask_view, do_abs)
    #     actual_idx, actual_val = fast_deconv.matrix.argmax(
    #         A_view, mask_view, do_abs, resources
    #     )
    #
    #     assert actual_idx == true_idx
    #     assert actual_val == true_val


@pytest.mark.parametrize("shape", [(16), (31, 17), (128, 64, 19), (2, 2, 5, 577), (100, 1, 1024, 1024)])
def test_subtract(shape):
    rng = cp.random.default_rng(12345)
    resources = fast_deconv.stream_resources()

    A = rng.standard_normal(shape, dtype=cp.float32)
    B = rng.standard_normal(shape, dtype=cp.float32)
    C = cp.zeros(shape, dtype=cp.float32)

    fast_deconv.matrix.subtract(A, B, C, resources)

    true_c = A - B

    cp.testing.assert_array_equal(C, true_c)
