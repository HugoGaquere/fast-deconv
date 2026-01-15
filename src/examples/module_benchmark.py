import cupy as cp
from benchmarks import benchmark, print_bench
import fast_deconv as fd
from functools import partial

rng = cp.random.default_rng(12345)
resources = fd.stream_resources()

shape = (8, 1, 1_000, 1_000)

A = rng.standard_normal(shape, dtype=cp.float32)
B = rng.standard_normal(shape, dtype=cp.float32)
C = cp.zeros(shape).astype(cp.float32)
mask = rng.random(shape) > 0.3

dirty = rng.standard_normal(shape, dtype=cp.float32)
psf = rng.standard_normal(shape, dtype=cp.float32)
coeffs = rng.standard_normal(shape[0], dtype=cp.float32)
gain = rng.standard_normal(1, dtype=cp.float32)
out_psf_dirty = cp.zeros(shape).astype(cp.float32)


print("==[ Module benchmark ]==")


cp.cuda.runtime.deviceSynchronize()


def argmax_cupy(data, mask, do_abs):
    data = cp.asarray(data)
    mask = cp.asarray(mask, dtype=cp.bool_)
    metric = cp.abs(data) if do_abs else data
    masked = cp.where(mask, metric, -cp.inf)
    flat_idx = int(cp.argmax(masked).item())
    orig_val = data.ravel()[flat_idx]
    ret_val = float(cp.abs(orig_val).item()) if do_abs else float(orig_val.item())
    cp.cuda.runtime.deviceSynchronize()
    return flat_idx, ret_val


print("[+] Running benchmarks ...")
bench = partial(benchmark, n_warmup=2, n_runs=10, n_iter=500)

print(f"Data: {A.shape=} {B.shape=} {mask.shape=}")

print("FastDeconv:")

bench_res = bench(fd.matrix.argmax, A, mask, False, resources)
print_bench(" argmax(A, mask)", bench_res)

bench_res = bench(fd.matrix.argmax, A, mask, True, resources)
print_bench(" argmax(abs(A), mask)", bench_res)

# bench_res = bench(fd.matrix.argmax, A[:, :3], mask[:, :3], False, resources)
# print_bench(" argmax_mdspan(A, mask)", bench_res)

bench_res = bench(fd.matrix.subtract, A, B, C, resources)
print_bench(" subtract(A, B)", bench_res)

bench_res = bench(fd.matrix.subtract, A[:, :30], B[:, :30], C[:, :30], resources)
print_bench(" subtract(A[:, :3], B[:, :3])", bench_res)

bench_res = bench(
    fd.wscms.subtract_psf_from_dirty,
    psf[..., 200:800, 200:800],
    dirty[..., 200:800, 200:800],
    coeffs,
    out_psf_dirty[..., 200:800, 200:800],
    gain,
    resources,
)
print_bench(" subtract_psf_from_dirty(A[:, :3], B[:, :3])", bench_res)

print("Cupy")
# argmax_cupy_bench = bench(argmax_cupy, A, mask, False)
# print_bench(" argmax(A, B)", argmax_cupy_bench)
#
# argmax_abs_cupy_bench = bench(argmax_cupy, A, mask, True)
# print_bench(" argmax(abs(A), mask)", argmax_abs_cupy_bench)
#
# subtract_cupy_bench = bench(lambda a, b, c: cp.subtract(a, b, out=c), A, B, C)
# print_bench(" subtract", subtract_cupy_bench)

def subtract_psf_from_dirty():
    scaled_psf = psf * coeffs[:, None, None, None] * gain
    cp.subtract(
        dirty[..., 200:800, 200:800],
        scaled_psf[..., 200:800, 200:800],
        out=out_psf_dirty[..., 200:800, 200:800],
    )
    cp.cuda.runtime.deviceSynchronize()
bench_res = bench(lambda : subtract_psf_from_dirty)
print_bench(" subtract_psf_from_dirty", bench_res )
# breakpoint()
