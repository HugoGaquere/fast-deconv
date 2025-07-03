import time
from functools import wraps

import cupy as cp
import nvtx
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


# @timeit
def bench(data, mask):
    with nvtx.annotate("argmax", color="green"):
        idx = kernels.argmax(data, mask)


@timeit
def bench_cupy(data, mask):
    masked_data = cp.where(mask, data, -cp.inf)
    argmax_index = cp.argmax(masked_data)


def test_argmax():
    size = 10_000_000
    data = cp.random.rand(size).astype(cp.float32)
    mask = cp.random.rand(size) > 0.5  # ~50% True, ~50% False

    masked_data = cp.where(mask, data, -cp.inf)
    argmax_index = cp.argmax(masked_data)

    kernels.argmax_init(size)
    idx = kernels.argmax(data, mask)
    assert argmax_index == idx
    kernels.argmax_free()


size = 10_000_000
data = cp.random.rand(size).astype(cp.float32)
mask = cp.random.rand(size) > 0.5  # ~50% True, ~50% False

with nvtx.annotate("argmax_init", color="blue"):
    kernels.argmax_init(size)
for _ in range(50):
    bench(data, mask)
kernels.argmax_free()

# bench_cupy(data, mask)
