"""
Benchmark cases for WSCMS (Weighted Stacked Clean Major/Minor Sweep) operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cupy as cp

import fast_deconv as fd

from .base import BenchmarkCase


@dataclass
class SubtractPsfFromDirtyBenchmark(BenchmarkCase):
    """
    Benchmark PSF subtraction from dirty image.

    Operation: out[k, 0, i, j] = dirty[k, 0, i, j] - psf[k, 0, i, j] * coeffs[k] * gain

    This is a core operation in CLEAN-based deconvolution algorithms.
    """

    n_channels: int = 16
    n_pol: int = 1  # Always 1 for this operation
    height: int = 512
    width: int = 512
    gain: float = 0.1
    dtype: cp.dtype = field(default_factory=lambda: cp.float32)

    name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self):
        self.name = "subtract_psf_from_dirty"
        self.description = (
            f"PSF subtraction ({self.n_channels}ch x {self.height}x{self.width})"
        )

    def setup(self) -> tuple[Callable[[], Any], Callable[[], Any]]:
        shape = (self.n_channels, self.n_pol, self.height, self.width)

        # Create 4D arrays with an extra element in last dim to create strided views
        # The WSCMS function requires layout_stride arrays
        padded_shape = (self.n_channels, self.n_pol, self.height, self.width + 1)
        psf_full = cp.random.randn(*padded_shape, dtype=self.dtype)
        dirty_full = cp.random.randn(*padded_shape, dtype=self.dtype)
        out_cupy_full = cp.empty_like(dirty_full)
        out_fd_full = cp.empty_like(dirty_full)

        # Create strided views (non-contiguous in last dimension)
        slc = (slice(None), slice(None), slice(None), slice(None, -1))
        psf = psf_full[slc]
        dirty = dirty_full[slc]
        out_cupy = out_cupy_full[slc]
        out_fd = out_fd_full[slc]

        coeffs = cp.random.randn(self.n_channels, dtype=self.dtype)

        gain = self.gain
        resources = fd.stream_resources()

        def cupy_fn():
            # CuPy equivalent: broadcasting coeffs across spatial dims
            # coeffs shape: (nch,) -> (nch, 1, 1, 1) for broadcasting
            coeffs_broadcast = coeffs[:, cp.newaxis, cp.newaxis, cp.newaxis]
            cp.subtract(
                dirty, psf * coeffs_broadcast * gain, out=out_cupy
            )
            return out_cupy

        def fast_deconv_fn():
            fd.wscms.subtract_psf_from_dirty_async(
                psf, dirty, coeffs, out_fd, gain, resources
            )
            cp.cuda.runtime.deviceSynchronize()
            return out_fd

        return cupy_fn, fast_deconv_fn

    def get_metadata(self) -> dict[str, Any]:
        return {
            "n_channels": self.n_channels,
            "n_pol": self.n_pol,
            "height": self.height,
            "width": self.width,
            "gain": self.gain,
            "dtype": str(self.dtype),
            "total_elements": self.n_channels * self.n_pol * self.height * self.width,
        }


@dataclass
class SubtractPsfFromDirtyStridedBenchmark(BenchmarkCase):
    """
    Benchmark PSF subtraction with strided (non-contiguous) arrays.

    This simulates real-world usage where PSF/dirty images are views
    into larger memory-mapped or pre-allocated buffers.
    """

    n_channels: int = 16
    n_pol: int = 1
    height: int = 512
    width: int = 512
    gain: float = 0.1
    stride_factor: int = 2
    dtype: cp.dtype = field(default_factory=lambda: cp.float32)

    name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self):
        self.name = "subtract_psf_from_dirty_strided"
        self.description = (
            f"PSF subtraction strided ({self.n_channels}ch x {self.height}x{self.width})"
        )

    def setup(self) -> tuple[Callable[[], Any], Callable[[], Any]]:
        # Create larger arrays and slice to get strided views
        full_shape = (
            self.n_channels * self.stride_factor,
            self.n_pol,
            self.height * self.stride_factor,
            self.width * self.stride_factor,
        )

        psf_full = cp.random.randn(*full_shape, dtype=self.dtype)
        dirty_full = cp.random.randn(*full_shape, dtype=self.dtype)
        out_full_cupy = cp.empty_like(dirty_full)
        out_full_fd = cp.empty_like(dirty_full)

        # Strided views
        sf = self.stride_factor
        psf = psf_full[::sf, :, ::sf, ::sf]
        dirty = dirty_full[::sf, :, ::sf, ::sf]
        out_cupy = out_full_cupy[::sf, :, ::sf, ::sf]
        out_fd = out_full_fd[::sf, :, ::sf, ::sf]

        coeffs = cp.random.randn(self.n_channels, dtype=self.dtype)
        gain = self.gain
        resources = fd.stream_resources()

        def cupy_fn():
            coeffs_broadcast = coeffs[:, cp.newaxis, cp.newaxis, cp.newaxis]
            cp.subtract(
                dirty, psf * coeffs_broadcast * gain, out=out_cupy
            )
            return out_cupy

        def fast_deconv_fn():
            fd.wscms.subtract_psf_from_dirty_async(
                psf, dirty, coeffs, out_fd, gain, resources
            )
            cp.cuda.runtime.deviceSynchronize()
            return out_fd

        return cupy_fn, fast_deconv_fn

    def get_metadata(self) -> dict[str, Any]:
        return {
            "n_channels": self.n_channels,
            "n_pol": self.n_pol,
            "height": self.height,
            "width": self.width,
            "gain": self.gain,
            "stride_factor": self.stride_factor,
            "dtype": str(self.dtype),
            "layout": "strided",
            "total_elements": self.n_channels * self.n_pol * self.height * self.width,
        }
