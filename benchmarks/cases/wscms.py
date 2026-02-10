"""
Benchmark cases for WSCMS (Weighted Stacked Clean Major/Minor Sweep) operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cupy as cp
import numpy as np

import fast_deconv as fd
import nvtx

from .base import BenchmarkCase


def _python_reference_minor_cycle(
    dirty, scaled_dirty, mask, psfs, psfs_2, gains, jones_norm,
    Xdes, sqrt_weights, map_pixels_facets,
    scale_idx, peak_factor, n_subminor_iter, do_abs, beam_enable,
):
    """Pure CuPy reference implementation of wscms_minor_cycle."""
    nch, npol, h, w = dirty.shape

    if do_abs:
        masked = cp.where(mask, cp.abs(scaled_dirty), 0)
    else:
        masked = cp.where(mask, scaled_dirty, -cp.inf)
    peak_flat = int(cp.argmax(masked).item())
    peak_coords = np.unravel_index(peak_flat, dirty.shape)
    peak_val = abs(float(scaled_dirty[peak_coords].item())) if do_abs else float(scaled_dirty[peak_coords].item())

    threshold = peak_factor * peak_val

    if do_abs:
        mask = mask & (cp.abs(dirty) > threshold)
    else:
        mask = mask & (dirty > threshold)

    sub_iter = 0
    while peak_val > threshold and sub_iter < n_subminor_iter:
        peak_ch, peak_pol, x, y = peak_coords
        facet_idx = int(map_pixels_facets[x, y].item())
        gain = float(gains[scale_idx, facet_idx].item())

        jn = jones_norm[:, 0, x, y]
        apparent_flux = dirty[:, 0, x, y]

        if beam_enable:
            SAX = cp.sqrt(jn)[:, None] * Xdes
        else:
            SAX = Xdes.copy()

        WX = (sqrt_weights[:, None] * SAX).astype(cp.float64)
        nchan, order = WX.shape
        if nchan >= order:
            pinv = cp.linalg.inv(WX.T @ WX) @ WX.T
        else:
            pinv = WX.T @ cp.linalg.inv(WX @ WX.T)

        Wy = (sqrt_weights * apparent_flux).astype(cp.float64)
        coeffs_compact = pinv @ Wy
        per_channel = (SAX.astype(cp.float64) @ coeffs_compact).astype(cp.float32)

        psf_h, psf_w = psfs.shape[4], psfs.shape[5]
        left = psf_w // 2; right = psf_w - left
        top = psf_h // 2; bottom = psf_h - top

        img_x0 = max(0, x - left); img_x1 = min(h, x + right)
        img_y0 = max(0, y - top); img_y1 = min(w, y + bottom)
        psf_x0 = img_x0 - (x - left); psf_x1 = psf_w - ((x + right) - img_x1)
        psf_y0 = img_y0 - (y - top); psf_y1 = psf_h - ((y + bottom) - img_y1)

        psf_sub = psfs[scale_idx, facet_idx, :, :, psf_x0:psf_x1, psf_y0:psf_y1]
        mask_region = mask[:, :, img_x0:img_x1, img_y0:img_y1].astype(cp.float32)
        dirty[:, :, img_x0:img_x1, img_y0:img_y1] -= (
            psf_sub * per_channel[:, None, None, None] * gain * mask_region
        )

        psf_2_sub = psfs_2[scale_idx, facet_idx, :, :, psf_x0:psf_x1, psf_y0:psf_y1]
        sd_peak = float(scaled_dirty[0, 0, x, y].item())
        gain_scaled = sd_peak * gain
        scaled_dirty[:, :, img_x0:img_x1, img_y0:img_y1] -= (
            psf_2_sub * gain_scaled * mask[:, :, img_x0:img_x1, img_y0:img_y1].astype(cp.float32)
        )

        if do_abs:
            masked = cp.where(mask, cp.abs(scaled_dirty), 0)
        else:
            masked = cp.where(mask, scaled_dirty, -cp.inf)
        peak_flat = int(cp.argmax(masked).item())
        peak_coords = np.unravel_index(peak_flat, dirty.shape)
        peak_val = abs(float(scaled_dirty[peak_coords].item())) if do_abs else float(scaled_dirty[peak_coords].item())
        sub_iter += 1


@dataclass
class WscmsMinorCycleBenchmark(BenchmarkCase):
    """
    Benchmark the full WSCMS sub-minor cycle loop.

    Compares the C++ wscms_minor_cycle against a pure CuPy reference
    implementation. The minor cycle iteratively finds peaks, performs
    spectral fitting, and subtracts PSFs from dirty/scaled_dirty images.
    """

    n_channels: int = 4
    n_pol: int = 1
    height: int = 64
    width: int = 64
    psf_height: int = 32
    psf_width: int = 32
    n_scales: int = 1
    n_facets: int = 1
    order: int = 2
    n_subminor_iter: int = 10
    peak_factor: float = 0.5
    do_abs: bool = True
    beam_enable: bool = True
    dtype: cp.dtype = field(default_factory=lambda: cp.float32)

    name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self):
        self.name = "wscms_minor_cycle"
        self.description = (
            f"Minor cycle ({self.n_channels}ch x {self.height}x{self.width}, "
            f"{self.n_subminor_iter} iters)"
        )

    def setup(self, stream: cp.cuda.Stream) -> tuple[Callable[[], Any], Callable[[], Any]]:
        rng = cp.random.default_rng(42)

        nch, npol, h, w = self.n_channels, self.n_pol, self.height, self.width
        psf_h, psf_w = self.psf_height, self.psf_width

        # Create initial data with enough signal for iterations
        dirty_init = rng.standard_normal((nch, npol, h, w), dtype=self.dtype) * 10
        scaled_dirty_init = dirty_init.copy()
        mask_init = cp.ones((nch, npol, h, w), dtype=cp.bool_)
        psfs = rng.standard_normal(
            (self.n_scales, self.n_facets, nch, npol, psf_h, psf_w), dtype=self.dtype
        )
        psfs_2 = rng.standard_normal(
            (self.n_scales, self.n_facets, nch, npol, psf_h, psf_w), dtype=self.dtype
        )
        gains = cp.ones((self.n_scales, self.n_facets), dtype=self.dtype) * 0.1
        jones_norm = cp.ones((nch, npol, h, w), dtype=self.dtype)
        Xdes = rng.standard_normal((nch, self.order), dtype=self.dtype)
        sqrt_weights = cp.ones(nch, dtype=self.dtype)
        map_pixels_facets = cp.zeros((h, w), dtype=cp.int32)

        # Working copies that get reset each call
        dirty_cupy = cp.empty_like(dirty_init)
        scaled_dirty_cupy = cp.empty_like(dirty_init)
        mask_cupy = cp.empty_like(mask_init)
        dirty_fd = cp.empty_like(dirty_init)
        scaled_dirty_fd = cp.empty_like(dirty_init)
        mask_fd = cp.empty_like(mask_init)

        resources = fd.stream_resources.from_cupy_stream(stream)

        ctx = fd.wscms.make_minor_cycle_context(
            jones_norm=jones_norm,
            map_pixels_facets=map_pixels_facets,
            Xdes=Xdes, sqrt_weights=sqrt_weights,
            beam_enable=self.beam_enable,
            peak_factor=self.peak_factor,
            n_subminor_iter=self.n_subminor_iter,
            do_abs=self.do_abs,
        )

        scale_idx = 0

        def cupy_fn():
            with stream:
                # Reset data
                cp.copyto(dirty_cupy, dirty_init)
                cp.copyto(scaled_dirty_cupy, scaled_dirty_init)
                cp.copyto(mask_cupy, mask_init)

                with nvtx.annotate("kernelsss"):
                    _python_reference_minor_cycle(
                        dirty_cupy, scaled_dirty_cupy, mask_cupy,
                        psfs, psfs_2, gains, jones_norm,
                        Xdes, sqrt_weights, map_pixels_facets,
                        scale_idx, self.peak_factor, self.n_subminor_iter,
                        self.do_abs, self.beam_enable,
                    )
            return dirty_cupy

        def fast_deconv_fn():
            # Reset data
            cp.copyto(dirty_fd, dirty_init)
            cp.copyto(scaled_dirty_fd, scaled_dirty_init)
            cp.copyto(mask_fd, mask_init)


            with nvtx.annotate("kernels"):
                fd.wscms.wscms_minor_cycle(
                    dirty_fd, scaled_dirty_fd, psfs, psfs_2, mask_fd, gains,
                    scale_idx, ctx, resources,
                )
            return dirty_fd

        return cupy_fn, fast_deconv_fn

    def _compare_results(self, cupy_result: Any, fast_deconv_result: Any) -> bool:
        """Allow looser tolerance for iterative algorithm."""
        return cp.allclose(cupy_result, fast_deconv_result, rtol=1e-4, atol=1e-4)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "n_channels": self.n_channels,
            "n_pol": self.n_pol,
            "height": self.height,
            "width": self.width,
            "psf_height": self.psf_height,
            "psf_width": self.psf_width,
            "n_scales": self.n_scales,
            "n_facets": self.n_facets,
            "order": self.order,
            "n_subminor_iter": self.n_subminor_iter,
            "peak_factor": self.peak_factor,
            "dtype": str(self.dtype),
            "total_elements": self.n_channels * self.n_pol * self.height * self.width,
        }
