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

    def setup(self, stream: cp.cuda.Stream) -> tuple[Callable[[], Any], Callable[[], Any]]:
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
        resources = fd.stream_resources.from_cupy_stream(stream)

        def cupy_fn():
            with stream:
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

    def setup(self, stream: cp.cuda.Stream) -> tuple[Callable[[], Any], Callable[[], Any]]:
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
        resources = fd.stream_resources.from_cupy_stream(stream)

        def cupy_fn():
            with stream:
                coeffs_broadcast = coeffs[:, cp.newaxis, cp.newaxis, cp.newaxis]
                cp.subtract(
                    dirty, psf * coeffs_broadcast * gain, out=out_cupy
                )
            return out_cupy

        def fast_deconv_fn():
            fd.wscms.subtract_psf_from_dirty_async(
                psf, dirty, coeffs, out_fd, gain, resources
            )
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


@dataclass
class CleanDirtiesBenchmark(BenchmarkCase):
    """
    Benchmark fused clean_dirties operation.

    Operations:
        dirty[i] -= psf[i] * coeffs[ch] * gain
        scaled_dirty[i] -= psf_2[i] * gain * mask[i]

    This is the core fused kernel for WSCMS deconvolution that performs
    both dirty and scaled_dirty subtraction in a single pass.
    """

    n_channels: int = 16
    n_pol: int = 1
    height: int = 512
    width: int = 512
    gain: float = 0.1
    mask_ratio: float = 0.5
    dtype: cp.dtype = field(default_factory=lambda: cp.float32)

    name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self):
        self.name = "clean_dirties"
        self.description = (
            f"Clean dirties fused ({self.n_channels}ch x {self.height}x{self.width})"
        )

    def setup(self, stream: cp.cuda.Stream) -> tuple[Callable[[], Any], Callable[[], Any]]:
        shape = (self.n_channels, self.n_pol, self.height, self.width)

        # Create padded arrays to get strided views (required by the kernel)
        padded_shape = (self.n_channels, self.n_pol, self.height, self.width + 1)

        psf_full = cp.random.randn(*padded_shape, dtype=self.dtype)
        psf_2_full = cp.random.randn(*padded_shape, dtype=self.dtype)
        dirty_cupy_full = cp.random.randn(*padded_shape, dtype=self.dtype)
        dirty_fd_full = dirty_cupy_full.copy()
        scaled_dirty_cupy_full = cp.random.randn(*padded_shape, dtype=self.dtype)
        scaled_dirty_fd_full = scaled_dirty_cupy_full.copy()
        mask_full = (cp.random.rand(*padded_shape) < self.mask_ratio).astype(self.dtype)

        # Strided views
        slc = (slice(None), slice(None), slice(None), slice(None, -1))
        psf = psf_full[slc]
        psf_2 = psf_2_full[slc]
        dirty_cupy = dirty_cupy_full[slc]
        dirty_fd = dirty_fd_full[slc]
        scaled_dirty_cupy = scaled_dirty_cupy_full[slc]
        scaled_dirty_fd = scaled_dirty_fd_full[slc]
        mask = mask_full[slc]

        coeffs = cp.random.randn(self.n_channels, dtype=self.dtype)
        gain = self.gain
        resources = fd.stream_resources.from_cupy_stream(stream)

        def cupy_fn():
            with stream:
                # Operation 1: dirty -= psf * coeffs * gain
                coeffs_broadcast = coeffs[:, cp.newaxis, cp.newaxis, cp.newaxis]
                dirty_cupy[...] -= psf * coeffs_broadcast * gain
                # Operation 2: scaled_dirty -= psf_2 * gain * mask
                scaled_dirty_cupy[...] -= psf_2 * gain * mask
            return (dirty_cupy, scaled_dirty_cupy)

        def fast_deconv_fn():
            fd.wscms.clean_dirties_async(
                psf, psf_2, dirty_fd, scaled_dirty_fd,
                coeffs, mask, gain, resources
            )
            return (dirty_fd, scaled_dirty_fd)

        return cupy_fn, fast_deconv_fn

    def _compare_results(self, cupy_result: Any, fast_deconv_result: Any) -> bool:
        """Compare both dirty and scaled_dirty outputs."""
        dirty_cupy, scaled_dirty_cupy = cupy_result
        dirty_fd, scaled_dirty_fd = fast_deconv_result
        return (
            cp.allclose(dirty_cupy, dirty_fd, rtol=1e-5, atol=1e-5) and
            cp.allclose(scaled_dirty_cupy, scaled_dirty_fd, rtol=1e-5, atol=1e-5)
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "n_channels": self.n_channels,
            "n_pol": self.n_pol,
            "height": self.height,
            "width": self.width,
            "gain": self.gain,
            "mask_ratio": self.mask_ratio,
            "dtype": str(self.dtype),
            "total_elements": self.n_channels * self.n_pol * self.height * self.width,
        }


@dataclass
class CleanDirtiesStridedBenchmark(BenchmarkCase):
    """
    Benchmark fused clean_dirties operation with strided arrays.

    This simulates real-world usage where images are views into larger
    pre-allocated buffers or memory-mapped files.
    """

    n_channels: int = 16
    n_pol: int = 1
    height: int = 512
    width: int = 512
    gain: float = 0.1
    mask_ratio: float = 0.5
    stride_factor: int = 2
    dtype: cp.dtype = field(default_factory=lambda: cp.float32)

    name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self):
        self.name = "clean_dirties_strided"
        self.description = (
            f"Clean dirties strided ({self.n_channels}ch x {self.height}x{self.width})"
        )

    def setup(self, stream: cp.cuda.Stream) -> tuple[Callable[[], Any], Callable[[], Any]]:
        # Create larger arrays and slice to get strided views
        sf = self.stride_factor
        full_shape = (
            self.n_channels * sf,
            self.n_pol,
            self.height * sf,
            self.width * sf,
        )

        psf_full = cp.random.randn(*full_shape, dtype=self.dtype)
        psf_2_full = cp.random.randn(*full_shape, dtype=self.dtype)
        dirty_cupy_full = cp.random.randn(*full_shape, dtype=self.dtype)
        dirty_fd_full = dirty_cupy_full.copy()
        scaled_dirty_cupy_full = cp.random.randn(*full_shape, dtype=self.dtype)
        scaled_dirty_fd_full = scaled_dirty_cupy_full.copy()
        mask_full = (cp.random.rand(*full_shape) < self.mask_ratio).astype(self.dtype)

        # Strided views
        psf = psf_full[::sf, :, ::sf, ::sf]
        psf_2 = psf_2_full[::sf, :, ::sf, ::sf]
        dirty_cupy = dirty_cupy_full[::sf, :, ::sf, ::sf]
        dirty_fd = dirty_fd_full[::sf, :, ::sf, ::sf]
        scaled_dirty_cupy = scaled_dirty_cupy_full[::sf, :, ::sf, ::sf]
        scaled_dirty_fd = scaled_dirty_fd_full[::sf, :, ::sf, ::sf]
        mask = mask_full[::sf, :, ::sf, ::sf]

        coeffs = cp.random.randn(self.n_channels, dtype=self.dtype)
        gain = self.gain
        resources = fd.stream_resources.from_cupy_stream(stream)

        def cupy_fn():
            with stream:
                coeffs_broadcast = coeffs[:, cp.newaxis, cp.newaxis, cp.newaxis]
                dirty_cupy[...] -= psf * coeffs_broadcast * gain
                scaled_dirty_cupy[...] -= psf_2 * gain * mask
            return (dirty_cupy, scaled_dirty_cupy)

        def fast_deconv_fn():
            fd.wscms.clean_dirties_async(
                psf, psf_2, dirty_fd, scaled_dirty_fd,
                coeffs, mask, gain, resources
            )
            return (dirty_fd, scaled_dirty_fd)

        return cupy_fn, fast_deconv_fn

    def _compare_results(self, cupy_result: Any, fast_deconv_result: Any) -> bool:
        """Compare both dirty and scaled_dirty outputs."""
        dirty_cupy, scaled_dirty_cupy = cupy_result
        dirty_fd, scaled_dirty_fd = fast_deconv_result
        return (
            cp.allclose(dirty_cupy, dirty_fd, rtol=1e-5, atol=1e-5) and
            cp.allclose(scaled_dirty_cupy, scaled_dirty_fd, rtol=1e-5, atol=1e-5)
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "n_channels": self.n_channels,
            "n_pol": self.n_pol,
            "height": self.height,
            "width": self.width,
            "gain": self.gain,
            "mask_ratio": self.mask_ratio,
            "stride_factor": self.stride_factor,
            "dtype": str(self.dtype),
            "layout": "strided",
            "total_elements": self.n_channels * self.n_pol * self.height * self.width,
        }
