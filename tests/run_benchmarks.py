import time
import cupy as cp
from functools import wraps
import fast_deconv

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
def bench_argmax(data, mask, resources):
    idx, value = fast_deconv.matrix.argmax(data, mask, False, resources)

@timeit
def bench_argmax_abs(data, mask, resources):
    idx, value = fast_deconv.matrix.argmax(data, mask, True, resources)

@timeit
def bench_argmax_cupy(data, mask):
    masked_data = cp.where(mask, data, 0)
    true_index = cp.argmax(masked_data)
    true_value = masked_data[true_index]

@timeit
def bench_argmax_abs_cupy(data, mask):
    masked_data = cp.where(mask, abs(data), 0)
    true_index = cp.argmax(masked_data)
    true_value = masked_data[true_index]

size = 10_000
data = cp.random.randn(size, size).astype(cp.float32)
mask = cp.random.rand(size, size) > 0.5  # ~50% ~50% False

resources = fast_deconv.stream_resources()

for _ in range(10):
    bench_argmax(data, mask, resources)
for _ in range(10):
    bench_argmax_abs(data, mask, resources)
for _ in range(10):
    bench_argmax_cupy(data, mask)
for _ in range(10):
    bench_argmax_abs_cupy(data, mask)
