import time
import cupy as cp
import numpy as np
from functools import wraps
import fast_deconv

def benchmark(fn, *args, n_warmup=5, n_runs=20, sync_gpu=True, **kwargs):
    # Warmup
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    return {
        "min": float(times.min()),
        "median": float(np.median(times)),
        "max": float(times.max()),
    }

def bench_argmax(data, mask, resources, n_iter=1000):
    for _ in range(n_iter):
        idx, value = fast_deconv.matrix.argmax(data, mask, False, resources)

def bench_argmax_abs(data, mask, resources, n_iter=1000):
    for _ in range(n_iter):
        idx, value = fast_deconv.matrix.argmax(data, mask, True, resources)

def bench_argmax_cupy(data, mask, n_iter=1000):
    for _ in range(n_iter):
        masked_data = cp.where(mask, data, 0)
        true_index = cp.argmax(masked_data)
        true_value = masked_data[true_index]

def bench_argmax_abs_cupy(data, mask, n_iter=1000):
    for _ in range(n_iter):
        masked_data = cp.where(mask, cp.abs(data), 0)
        true_index = cp.argmax(masked_data)
        true_value = masked_data[true_index]

size = 10_000
data = cp.random.randn(size, size).astype(cp.float32)
mask = cp.random.rand(size, size) > 0.5  # bool mask on GPU

resources = fast_deconv.stream_resources()

N_ITER = 1000

res_fd = benchmark(bench_argmax, data, mask, resources, n_runs=1, n_warmup=3, sync_gpu=True, n_iter=N_ITER)
res_fd_abs = benchmark(bench_argmax_abs, data, mask, resources, n_runs=1, n_warmup=3, sync_gpu=True, n_iter=N_ITER)
res_cp = benchmark(bench_argmax_cupy, data, mask, n_runs=1, n_warmup=3, sync_gpu=True, n_iter=N_ITER)
res_cp_abs = benchmark(bench_argmax_abs_cupy, data, mask, n_runs=1, n_warmup=3, sync_gpu=True, n_iter=N_ITER)

def pretty(name, stats):
    per_iter = stats["median"] / N_ITER
    print(f"{name:20s} total median: {stats['median']:.6f} s  -> {per_iter*1e6:.2f} µs / call")

pretty("fast_deconv argmax", res_fd)
pretty("fast_deconv abs",    res_fd_abs)
pretty("cupy argmax",        res_cp)
pretty("cupy abs",           res_cp_abs)

