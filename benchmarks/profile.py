"""
Profiling script optimized for NVIDIA Nsight Compute and Nsight Systems.

Usage with Nsight Systems (timeline view):
    nsys profile -o profile_output python -m benchmarks.profile
    nsys profile -o profile_output python -m benchmarks.profile --case argmax
    nsys-ui profile_output.nsys-rep

Usage with Nsight Compute (kernel analysis):
    ncu --set full -o ncu_output python -m benchmarks.profile --case subtract
    ncu --set full --kernel-name subtract python -m benchmarks.profile
    ncu-ui ncu_output.ncu-rep

Useful ncu options:
    --kernel-name <regex>     Profile only matching kernels
    --launch-skip <n>         Skip first n kernel launches (warmup)
    --launch-count <n>        Profile only n kernel launches
    --set full                Full metrics collection
    --set roofline            Roofline analysis
    --set memory              Memory throughput analysis
"""

from __future__ import annotations

import argparse
import sys

import cupy as cp

import fast_deconv as fd
from benchmarks.core import nvtx_range


def profile_argmax():
    """Profile argmax operation."""
    print("Profiling: argmax")

    shape = (1024, 1024)
    data = cp.random.randn(*shape, dtype=cp.float32)
    mask = cp.ones(shape, dtype=cp.bool_)
    resources = fd.stream_resources()

    # Warmup
    for _ in range(5):
        fd.matrix.argmax(data, mask, False, resources)
    cp.cuda.Stream.null.synchronize()

    # Profiled iterations
    for i in range(10):
        with nvtx_range(f"argmax_iter_{i}", color="blue"):
            fd.matrix.argmax(data, mask, False, resources)

    cp.cuda.Stream.null.synchronize()


def profile_argmax_abs():
    """Profile argmax with absolute values."""
    print("Profiling: argmax_abs")

    shape = (1024, 1024)
    data = cp.random.randn(*shape, dtype=cp.float32) * 10
    mask = cp.ones(shape, dtype=cp.bool_)
    resources = fd.stream_resources()

    # Warmup
    for _ in range(5):
        fd.matrix.argmax(data, mask, True, resources)
    cp.cuda.Stream.null.synchronize()

    # Profiled iterations
    for i in range(10):
        with nvtx_range(f"argmax_abs_iter_{i}", color="green"):
            fd.matrix.argmax(data, mask, True, resources)

    cp.cuda.Stream.null.synchronize()


def profile_subtract():
    """Profile subtract operation."""
    print("Profiling: subtract")

    shape = (2048, 2048)
    a = cp.random.randn(*shape, dtype=cp.float32)
    b = cp.random.randn(*shape, dtype=cp.float32)
    c = cp.empty_like(a)
    resources = fd.stream_resources()

    # Warmup
    for _ in range(5):
        fd.matrix.subtract(a, b, c, resources)
    cp.cuda.Stream.null.synchronize()

    # Profiled iterations
    for i in range(10):
        with nvtx_range(f"subtract_iter_{i}", color="orange"):
            fd.matrix.subtract(a, b, c, resources)

    cp.cuda.Stream.null.synchronize()


def profile_subtract_strided():
    """Profile subtract with strided arrays."""
    print("Profiling: subtract_strided")

    full_shape = (4096, 4096)
    a_full = cp.random.randn(*full_shape, dtype=cp.float32)
    b_full = cp.random.randn(*full_shape, dtype=cp.float32)
    c_full = cp.empty_like(a_full)

    # Strided views
    a = a_full[::2, ::2]
    b = b_full[::2, ::2]
    c = c_full[::2, ::2]
    resources = fd.stream_resources()

    # Warmup
    for _ in range(5):
        fd.matrix.subtract(a, b, c, resources)
    cp.cuda.Stream.null.synchronize()

    # Profiled iterations
    for i in range(10):
        with nvtx_range(f"subtract_strided_iter_{i}", color="red"):
            fd.matrix.subtract(a, b, c, resources)

    cp.cuda.Stream.null.synchronize()


def profile_subtract_psf_from_dirty():
    """Profile WSCMS PSF subtraction."""
    print("Profiling: subtract_psf_from_dirty")

    n_channels = 16
    height = 1024
    width = 1024

    # Create padded arrays and slice to get strided views (required by the function)
    padded_shape = (n_channels, 1, height, width + 1)
    psf_full = cp.random.randn(*padded_shape, dtype=cp.float32)
    dirty_full = cp.random.randn(*padded_shape, dtype=cp.float32)
    out_full = cp.empty_like(dirty_full)

    # Strided views
    slc = (slice(None), slice(None), slice(None), slice(None, -1))
    psf = psf_full[slc]
    dirty = dirty_full[slc]
    out = out_full[slc]

    coeffs = cp.random.randn(n_channels, dtype=cp.float32)
    gain = 0.1
    resources = fd.stream_resources()

    # Warmup
    for _ in range(5):
        fd.wscms.subtract_psf_from_dirty_async(psf, dirty, coeffs, out, gain, resources)
        cp.cuda.runtime.deviceSynchronize()
    cp.cuda.Stream.null.synchronize()

    # Profiled iterations
    for i in range(10):
        with nvtx_range(f"subtract_psf_from_dirty_iter_{i}", color="purple"):
            fd.wscms.subtract_psf_from_dirty_async(
                psf, dirty, coeffs, out, gain, resources
            )
            cp.cuda.runtime.deviceSynchronize()

    cp.cuda.Stream.null.synchronize()


def profile_all():
    """Profile all operations."""
    with nvtx_range("all_benchmarks", color="white"):
        profile_argmax()
        profile_argmax_abs()
        profile_subtract()
        profile_subtract_strided()
        profile_subtract_psf_from_dirty()


PROFILE_CASES = {
    "argmax": profile_argmax,
    "argmax_abs": profile_argmax_abs,
    "subtract": profile_subtract,
    "subtract_strided": profile_subtract_strided,
    "subtract_psf_from_dirty": profile_subtract_psf_from_dirty,
    "all": profile_all,
}


def main():
    parser = argparse.ArgumentParser(
        description="Profile fast-deconv operations for Nsight",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--case",
        "-c",
        choices=list(PROFILE_CASES.keys()),
        default="all",
        help="Which operation to profile (default: all)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available profile cases",
    )

    args = parser.parse_args()

    if args.list:
        print("Available profile cases:")
        for case in PROFILE_CASES:
            print(f"  - {case}")
        return 0

    print(f"Running profile case: {args.case}")
    print("=" * 50)

    PROFILE_CASES[args.case]()

    print("=" * 50)
    print("Profiling complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
