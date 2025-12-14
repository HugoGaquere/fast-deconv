import time
import cupy as cp
import numpy as np


def benchmark(fn, *args, n_warmup=5, n_runs=20, n_iter=1000):
    for _ in range(n_warmup):
        fn(*args)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for _ in range(n_iter):
            fn(*args)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    return {
        "min": float(times.min()),
        "median": float(np.median(times)),
        "max": float(times.max()),
        "per_iter": float(np.median(times)) / n_iter,
        "n_iter": n_iter,
    }


def print_bench(name, stats):
    n_iter = stats['n_iter']
    per_iter = stats['median'] / n_iter
    print(
        f"{name:20s} total median ({n_iter} iter): {stats['median']:.6f} s  -> {per_iter * 1e6:.2f} µs / call"
    )
