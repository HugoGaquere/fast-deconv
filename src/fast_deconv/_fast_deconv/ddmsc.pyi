"""
DDMSC module
"""
from __future__ import annotations
import enum
import typing
import numpy as np
import numpy.typing as npt
__all__: list[str] = ['ConvergenceStatus', 'DDMSC', 'DDMSCResult']
class ConvergenceStatus(enum.Enum):
    running = 0
    converged = 1
    diverged = 2
    max_iterations = 3
    all_scales_stalled = 4
    no_components = 5
class DDMSC:
    auto_mask: bool
    clean_negative: bool
    force_auto_mask: bool
    def __init__(
        self,
        raw_psfs: npt.NDArray[np.float32],
        xdes: npt.NDArray[np.float32],
        scale_mask: npt.NDArray[np.bool_],
        scale_sigmas: npt.NDArray[np.float32],
        scale_bias: npt.NDArray[np.float32],
        map_pixel_facet: npt.NDArray[np.int32],
        dirty_nrow: int,
        dirty_ncol: int,
        n_freq: int,
        fft_padding: float,
        exec_device: int = ...,
    ) -> None:
        ...
    def run(
        self,
        dirty: npt.NDArray[np.float32],
        jones_norm: npt.NDArray[np.float32],
        weights_freq: npt.NDArray[np.float32],
    ) -> DDMSCResult:
        """
        Run the DDMSC minor-cycle loop with stall/divergence checks.
        """
    @property
    def auto_mask_peak_threshold(self) -> float | None:
        ...
    @auto_mask_peak_threshold.setter
    def auto_mask_peak_threshold(self, arg1: typing.SupportsFloat | typing.SupportsIndex | None) -> None:
        ...
    @property
    def auto_mask_rms_threshold(self) -> float | None:
        ...
    @auto_mask_rms_threshold.setter
    def auto_mask_rms_threshold(self, arg1: typing.SupportsFloat | typing.SupportsIndex | None) -> None:
        ...
    @property
    def divergence_factor(self) -> float:
        ...
    @divergence_factor.setter
    def divergence_factor(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def flux_threshold(self) -> float:
        ...
    @flux_threshold.setter
    def flux_threshold(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gamma(self) -> float:
        ...
    @gamma.setter
    def gamma(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_iteration(self) -> int:
        ...
    @max_iteration.setter
    def max_iteration(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_sub_iteration(self) -> int:
        ...
    @max_sub_iteration.setter
    def max_sub_iteration(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def peak_factor(self) -> float:
        ...
    @peak_factor.setter
    def peak_factor(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def stall_threshold(self) -> float:
        ...
    @stall_threshold.setter
    def stall_threshold(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def stop_cycle_factor(self) -> float:
        ...
    @stop_cycle_factor.setter
    def stop_cycle_factor(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def stop_peak_factor(self) -> float:
        ...
    @stop_peak_factor.setter
    def stop_peak_factor(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def stop_rms_factor(self) -> float:
        ...
    @stop_rms_factor.setter
    def stop_rms_factor(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def stop_sidelobe_level(self) -> float:
        ...
    @stop_sidelobe_level.setter
    def stop_sidelobe_level(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class DDMSCResult:
    @property
    def coeffs(self) -> list[list[float]]:
        ...
    @property
    def final_flux(self) -> float:
        ...
    @property
    def gains(self) -> list[float]:
        ...
    @property
    def peak_coords(self) -> list[tuple[int, int]]:
        ...
    @property
    def scales(self) -> list[int]:
        ...
    @property
    def status(self) -> ConvergenceStatus:
        ...
    @property
    def stop_flux(self) -> float:
        ...
    @property
    def total_iterations(self) -> int:
        ...
