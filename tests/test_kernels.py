import time
from functools import wraps

import cupy as cp
from fast_deconv import kernels


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[{func.__name__}] Execution time: {end - start:.6f} seconds")
        return result

    return wrapper


@timeit
def bench_argmax(data, mask):
    # with nvtx.annotate("argmax_f", color="green"):
    idx = kernels.argmax(data, mask)


@timeit
def bench_argmax_cupy(data, mask):
    masked_data = cp.where(mask, data, -cp.inf)
    argmax_index = cp.argmax(masked_data)


def test_argmax():
    size = 10_000_000
    data = cp.random.rand(size).astype(cp.float32)
    mask = cp.random.rand(size) > 0.5  # ~50% True, ~50% False

    masked_data = cp.where(mask, data, -cp.inf)
    argmax_index = cp.argmax(masked_data)

    idx = kernels.argmax(data, mask)
    assert argmax_index == idx


# size = 10_000_000
# data = cp.random.rand(size).astype(cp.float32)
# mask = cp.random.rand(size) > 0.5  # ~50% True, ~50% False
# # masked_data = cp.where(mask, data, -cp.inf)
#
# # with nvtx.annotate("argmax_f_init", color="blue"):
# # kernels.argmax_initF(size)
# for _ in range(10):
#     bench_f(data, mask)
# for _ in range(10):
#     bench_cupy(data, mask)
