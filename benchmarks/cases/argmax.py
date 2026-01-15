"""
Benchmark cases for argmax operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cupy as cp

import fast_deconv as fd

from .base import BenchmarkCase


@dataclass
class ArgmaxBenchmark(BenchmarkCase):
    """Benchmark argmax without mask."""

    shape: tuple[int, ...] = (1024, 1024)
    dtype: cp.dtype = field(default_factory=lambda: cp.float32)

    name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self):
        self.name = f"argmax_{len(self.shape)}d"
        self.description = f"Find maximum element in {len(self.shape)}D array"

    def setup(self) -> tuple[Callable[[], Any], Callable[[], Any]]:
        # Generate random data
        data = cp.random.randn(*self.shape, dtype=self.dtype)
        # Full mask (all True = no masking)
        mask = cp.ones(self.shape, dtype=cp.bool_)

        resources = fd.stream_resources()

        def cupy_fn():
            idx = int(cp.argmax(data))
            val = float(data.flat[idx])
            return idx, val

        def fast_deconv_fn():
            return fd.matrix.argmax(data, mask, False, resources)

        return cupy_fn, fast_deconv_fn

    def get_metadata(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "dtype": str(self.dtype),
            "total_elements": int(cp.prod(cp.array(self.shape))),
        }


@dataclass
class ArgmaxAbsBenchmark(BenchmarkCase):
    """Benchmark argmax with absolute values."""

    shape: tuple[int, ...] = (1024, 1024)
    dtype: cp.dtype = field(default_factory=lambda: cp.float32)

    name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self):
        self.name = f"argmax_abs_{len(self.shape)}d"
        self.description = f"Find maximum absolute value in {len(self.shape)}D array"

    def setup(self) -> tuple[Callable[[], Any], Callable[[], Any]]:
        # Generate random data with negative values
        data = cp.random.randn(*self.shape, dtype=self.dtype) * 10
        mask = cp.ones(self.shape, dtype=cp.bool_)

        resources = fd.stream_resources()

        def cupy_fn():
            abs_data = cp.abs(data)
            idx = int(cp.argmax(abs_data))
            val = float(abs_data.flat[idx])  # Return absolute value to match fast_deconv
            return idx, val

        def fast_deconv_fn():
            return fd.matrix.argmax(data, mask, True, resources)

        return cupy_fn, fast_deconv_fn

    def get_metadata(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "dtype": str(self.dtype),
            "use_abs": True,
            "total_elements": int(cp.prod(cp.array(self.shape))),
        }


@dataclass
class MaskedArgmaxBenchmark(BenchmarkCase):
    """Benchmark argmax with masking."""

    shape: tuple[int, ...] = (1024, 1024)
    mask_ratio: float = 0.5  # Fraction of elements to mask out
    dtype: cp.dtype = field(default_factory=lambda: cp.float32)

    name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self):
        self.name = f"argmax_masked_{len(self.shape)}d"
        self.description = (
            f"Find maximum in {len(self.shape)}D array with {self.mask_ratio:.0%} masked"
        )

    def setup(self) -> tuple[Callable[[], Any], Callable[[], Any]]:
        data = cp.random.randn(*self.shape, dtype=self.dtype)
        # Random mask
        mask = cp.random.rand(*self.shape) > self.mask_ratio

        resources = fd.stream_resources()

        def cupy_fn():
            # CuPy approach: set masked values to -inf
            masked_data = cp.where(mask, data, -cp.inf)
            idx = int(cp.argmax(masked_data))
            val = float(data.flat[idx])
            return idx, val

        def fast_deconv_fn():
            return fd.matrix.argmax(data, mask, False, resources)

        return cupy_fn, fast_deconv_fn

    def get_metadata(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "dtype": str(self.dtype),
            "mask_ratio": self.mask_ratio,
            "total_elements": int(cp.prod(cp.array(self.shape))),
        }
