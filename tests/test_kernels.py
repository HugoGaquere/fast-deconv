import time
from functools import wraps

import cupy as cp
import numpy as np
from fast_deconv import kernels

def test_argmax():
    for _ in range(100):
        size = 10_000_000
        data = cp.random.randn(size).astype(cp.float32)
        mask = cp.random.rand(size) > 0.5  # ~50% True, ~50% False

        masked_data = cp.where(mask, data, -cp.inf)
        true_index = cp.argmax(masked_data)
        true_value = data[true_index]

        actual_index, actual_value = kernels.argmax(data, mask, False)

        assert true_index == actual_index
        assert true_value == actual_value


def test_argmax_abs():
    for _ in range(100):
        size = 10_000_000
        data = cp.random.randn(size).astype(cp.float32)
        mask = cp.random.rand(size) > 0.5  # ~50% ~50% False

        masked_data = cp.where(mask, data, 0)
        true_index = cp.argmax(abs(masked_data))
        true_value = abs(data[true_index])

        actual_index, actual_value = kernels.argmax(data, mask, True)

        assert true_index == actual_index
        assert true_value == actual_value

# def timeit(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start = time.perf_counter()
#         result = func(*args, **kwargs)
#         end = time.perf_counter()
#         print(f"[{func.__name__}] Execution time: {end - start:.6f} seconds")
#         return result
#
#     return wrapper


# @timeit
# def bench_argmax(data, mask):
#     # with nvtx.annotate("argmax_f", color="green"):
#     idx = kernels.argmax(data, mask)


# @timeit
# def bench_argmax_cupy(data, mask):
#     masked_data = cp.where(mask, data, -cp.inf)
#     argmax_index = cp.argmax(masked_data)


# @timeit
# def bench_argmax_numpy(data, mask):
#     masked_data = np.where(mask, data, -np.inf)
#     argmax_index = np.argmax(masked_data)
# size = 10_000_000
# data = cp.random.randn(size).astype(cp.float32)
# mask = cp.random.rand(size) > 0.5  # ~50% ~50% False
# breakpoint()

# with nvtx.annotate("argmax_f_init", color="blue"):
# for _ in range(10):
#     bench_argmax(data, mask)
# for _ in range(10):
#     bench_argmax_cupy(data, mask)
# data = np.random.rand(size).astype(np.float32)
# mask = np.random.rand(size) > 0.5  # ~50% True, ~50% False
# for _ in range(10):
#     bench_argmax_numpy(data, mask)
