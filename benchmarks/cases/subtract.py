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

    def setup(self, stream: cp.cuda.Stream) -> tuple[Callable[[], Any], Callable[[], Any]]:
        # Contiguous arrays (C-order / row-major)
        a = cp.random.randn(*self.shape, dtype=self.dtype)
        b = cp.random.randn(*self.shape, dtype=self.dtype)
        c_cupy = cp.empty_like(a)
        c_fd = cp.empty_like(a)

        resources = fd.stream_resources.from_cupy_stream(stream)

        def cupy_fn():
            with stream:
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
    """Benchmark element-wise subtraction with strided views (contiguous inner stride).

    Creates subviews of larger arrays to simulate strided access patterns common in
    real applications. The inner stride remains contiguous (stride=1) while outer
    dimensions have non-unit strides due to the padding.

    This tests performance with memory layouts where:
    - Data is not contiguous in memory overall
    - But the innermost dimension IS contiguous (cache-friendly access pattern)
    """

    view_shape: tuple[int, ...] = (1024, 1024)
    padding: int = 16  # Padding on each side to create non-unit outer strides
    dtype: cp.dtype = field(default_factory=lambda: cp.float32)
    label: str = ""  # Optional label for identifying aspect ratio (e.g., "wide", "tall")

    name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self):
        ndim = len(self.view_shape)
        label_suffix = f"_{self.label}" if self.label else ""
        self.name = f"subtract_strided_{ndim}d{label_suffix}"

        # Create shape description
        shape_str = "x".join(str(s) for s in self.view_shape)
        aspect_info = f" ({self.label})" if self.label else ""
        self.description = (
            f"Element-wise subtract for {ndim}D strided view [{shape_str}]{aspect_info} "
            f"(padding={self.padding}, contiguous inner stride)"
        )

    def setup(self, stream: cp.cuda.Stream) -> tuple[Callable[[], Any], Callable[[], Any]]:
        # Create larger arrays with padding for offset slicing
        # Only pad outer dimensions to maintain contiguous inner stride
        # For row-major (C-order), the last dimension is the innermost
        full_shape = tuple(s + 2 * self.padding for s in self.view_shape)
        a_full = cp.random.randn(*full_shape, dtype=self.dtype)
        b_full = cp.random.randn(*full_shape, dtype=self.dtype)
        c_full_cupy = cp.empty_like(a_full)
        c_full_fd = cp.empty_like(a_full)

        # Create strided views: slice [padding:padding+size] for each dimension
        slices = tuple(slice(self.padding, self.padding + s) for s in self.view_shape)
        a = a_full[slices]
        b = b_full[slices]
        c_cupy = c_full_cupy[slices]
        c_fd = c_full_fd[slices]

        resources = fd.stream_resources.from_cupy_stream(stream)

        def cupy_fn():
            with stream:
                cp.subtract(a, b, out=c_cupy)
            return c_cupy

        def fast_deconv_fn():
            fd.matrix.subtract(a, b, c_fd, resources)
            return c_fd

        return cupy_fn, fast_deconv_fn

    def get_metadata(self) -> dict[str, Any]:
        total_elements = int(cp.prod(cp.array(self.view_shape)))
        full_shape = tuple(s + 2 * self.padding for s in self.view_shape)

        # Calculate actual strides of the view (in elements, not bytes)
        # For a C-order array, stride[i] = product of shape[i+1:]
        strides = []
        stride = 1
        for s in reversed(full_shape):
            strides.append(stride)
            stride *= s
        strides = tuple(reversed(strides))

        return {
            "view_shape": self.view_shape,
            "full_shape": full_shape,
            "dtype": str(self.dtype),
            "layout": "strided_contiguous_inner",
            "padding": self.padding,
            "strides": strides,
            "total_elements": total_elements,
            "label": self.label,
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

    def setup(self, stream: cp.cuda.Stream) -> tuple[Callable[[], Any], Callable[[], Any]]:
        # We need fresh copies for each iteration to keep results consistent
        a_orig = cp.random.randn(*self.shape, dtype=self.dtype)
        b = cp.random.randn(*self.shape, dtype=self.dtype)

        # Pre-allocate working copies
        a_cupy = cp.empty_like(a_orig)
        a_fd = cp.empty_like(a_orig)

        resources = fd.stream_resources.from_cupy_stream(stream)

        def cupy_fn():
            with stream:
                cp.copyto(a_cupy, a_orig)
                cp.subtract(a_cupy, b, out=a_cupy)
            return a_cupy

        def fast_deconv_fn():
            with stream:
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
