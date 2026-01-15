"""
Benchmark cases for subtract operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cupy as cp

import fast_deconv as fd

from .base import BenchmarkCase


@dataclass
class SubtractBenchmark(BenchmarkCase):
    """Benchmark element-wise subtraction with contiguous arrays."""

    shape: tuple[int, ...] = (1024, 1024)
    dtype: cp.dtype = field(default_factory=lambda: cp.float32)

    name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self):
        self.name = f"subtract_{len(self.shape)}d"
        self.description = f"Element-wise subtract for {len(self.shape)}D contiguous arrays"

    def setup(self) -> tuple[Callable[[], Any], Callable[[], Any]]:
        # Contiguous arrays (C-order / row-major)
        a = cp.random.randn(*self.shape, dtype=self.dtype)
        b = cp.random.randn(*self.shape, dtype=self.dtype)
        c_cupy = cp.empty_like(a)
        c_fd = cp.empty_like(a)

        resources = fd.stream_resources()

        def cupy_fn():
            cp.subtract(a, b, out=c_cupy)
            return c_cupy

        def fast_deconv_fn():
            fd.matrix.subtract(a, b, c_fd, resources)
            return c_fd

        return cupy_fn, fast_deconv_fn

    def get_metadata(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "dtype": str(self.dtype),
            "layout": "contiguous",
            "total_elements": int(cp.prod(cp.array(self.shape))),
        }


@dataclass
class SubtractStridedBenchmark(BenchmarkCase):
    """Benchmark element-wise subtraction with strided (non-contiguous) arrays."""

    shape: tuple[int, ...] = (1024, 1024)
    stride_factor: int = 2  # Every nth element
    dtype: cp.dtype = field(default_factory=lambda: cp.float32)

    name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self):
        self.name = f"subtract_strided_{len(self.shape)}d"
        self.description = (
            f"Element-wise subtract for {len(self.shape)}D strided arrays "
            f"(stride={self.stride_factor})"
        )

    def setup(self) -> tuple[Callable[[], Any], Callable[[], Any]]:
        # Create larger arrays and slice to get strided views
        full_shape = tuple(s * self.stride_factor for s in self.shape)
        a_full = cp.random.randn(*full_shape, dtype=self.dtype)
        b_full = cp.random.randn(*full_shape, dtype=self.dtype)
        c_full_cupy = cp.empty_like(a_full)
        c_full_fd = cp.empty_like(a_full)

        # Create strided views (non-contiguous)
        slices = tuple(slice(None, None, self.stride_factor) for _ in self.shape)
        a = a_full[slices]
        b = b_full[slices]
        c_cupy = c_full_cupy[slices]
        c_fd = c_full_fd[slices]

        resources = fd.stream_resources()

        def cupy_fn():
            cp.subtract(a, b, out=c_cupy)
            return c_cupy

        def fast_deconv_fn():
            fd.matrix.subtract(a, b, c_fd, resources)
            return c_fd

        return cupy_fn, fast_deconv_fn

    def get_metadata(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "dtype": str(self.dtype),
            "layout": "strided",
            "stride_factor": self.stride_factor,
            "total_elements": int(cp.prod(cp.array(self.shape))),
        }


@dataclass
class SubtractInPlaceBenchmark(BenchmarkCase):
    """Benchmark in-place subtraction (A = A - B)."""

    shape: tuple[int, ...] = (1024, 1024)
    dtype: cp.dtype = field(default_factory=lambda: cp.float32)

    name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self):
        self.name = f"subtract_inplace_{len(self.shape)}d"
        self.description = f"In-place subtract for {len(self.shape)}D arrays"

    def setup(self) -> tuple[Callable[[], Any], Callable[[], Any]]:
        # We need fresh copies for each iteration to keep results consistent
        a_orig = cp.random.randn(*self.shape, dtype=self.dtype)
        b = cp.random.randn(*self.shape, dtype=self.dtype)

        # Pre-allocate working copies
        a_cupy = cp.empty_like(a_orig)
        a_fd = cp.empty_like(a_orig)

        resources = fd.stream_resources()

        def cupy_fn():
            cp.copyto(a_cupy, a_orig)
            cp.subtract(a_cupy, b, out=a_cupy)
            return a_cupy

        def fast_deconv_fn():
            cp.copyto(a_fd, a_orig)
            fd.matrix.subtract(a_fd, b, a_fd, resources)
            return a_fd

        return cupy_fn, fast_deconv_fn

    def get_metadata(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "dtype": str(self.dtype),
            "operation": "in_place",
            "total_elements": int(cp.prod(cp.array(self.shape))),
        }
