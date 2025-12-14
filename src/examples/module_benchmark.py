import cupy as cp
import numpy as np
from benchmarks import benchmark, print_bench
import fast_deconv as fd
from functools import partial

rng = cp.random.default_rng(12345)
rng_h = np.random.default_rng(12345)
resources = fd.stream_resources()

shape = (1, 1, 5, 5)

A = rng.standard_normal(shape, dtype=cp.float32)
B = rng.standard_normal(shape, dtype=cp.float32)
C = cp.zeros(shape).astype(cp.float32)
mask = rng.random(shape) > 0.3


print(A)
print(B)

breakpoint()
fd.matrix.subtract(A, B, C, resources)

cp.testing.assert_array_equal(C, A - B)





if False:
    mask = cp.ones(shape, bool)

    A_v = A[:, :2]
    mask_v = mask[:, :2]
# fd.print_host(A_h[0])

    print("A")
    print(A)
    print("A[:, :2]")
    print(A_v)
    print("Mdspan Host view: A[:, :2]")
# fd.print_host(A_v.get())
    print("Mdspan Device view: A[:, :2]")
    fd.print(A_v)
    breakpoint()

# m = fd.matrix.argmax(A_v, mask_v, True, resources)
# print(f"{m=}")


print("==[ Module benchmark ]==")

def argmax_cupy(data, mask, do_abs):
    data = cp.asarray(data)
    mask = cp.asarray(mask, dtype=cp.bool_)
    metric = cp.abs(data) if do_abs else data
    masked = cp.where(mask, metric, -cp.inf)
    flat_idx = int(cp.argmax(masked).item())
    orig_val = data.ravel()[flat_idx]
    ret_val = float(cp.abs(orig_val).item()) if do_abs else float(orig_val.item())
    return flat_idx, ret_val


print("[+] Running benchmarks ...")
bench = partial(benchmark, n_warmup=2, n_runs=10, n_iter=500)

print(f"Data: {A.shape=} {B.shape=} {mask.shape=}")

print("FastDeconv:")

bench_res = bench(fd.matrix.argmax, A, mask, False, resources)
print_bench(" argmax(A, mask)", bench_res)

bench_res = bench(fd.matrix.argmax, A, mask, True, resources)
print_bench(" argmax(abs(A), mask)", bench_res)

bench_res = bench(fd.matrix.argmax_mdspan, A, mask, False, resources)
print_bench(" argmax_mdspan(A, mask)", bench_res)

bench_res = bench(fd.matrix.argmax_mdspan, A, mask, True, resources)
print_bench(" argmax_mdspan(abs(A), mask)", bench_res)

bench_res = bench(fd.matrix.subtract, A, B, C, resources)
print_bench(" subtract(A, B)", bench_res)

exit(0)

print("Cupy")
argmax_cupy_bench = bench(argmax_cupy, A, mask, False)
print_bench(" argmax(A, B)", argmax_cupy_bench)

argmax_abs_cupy_bench = bench(argmax_cupy, A, mask, True)
print_bench(" argmax(abs(A), mask)", argmax_abs_cupy_bench)

subtract_cupy_bench = bench(lambda a, b, c: cp.subtract(a, b, out=c), A, B, C)
print_bench(" subtract", subtract_cupy_bench)

# breakpoint()
