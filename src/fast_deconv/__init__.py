import os

import numpy as np

from fast_deconv._fast_deconv import ddmsc

DDMSCResult = ddmsc.DDMSCResult
ConvergenceStatus = ddmsc.ConvergenceStatus

# Every tunable property, snapshotted per run() so the replay needs no
# knowledge of which ones the caller changes between cycles.
_DUMP_PARAMS = (
    "clean_negative",
    "peak_factor",
    "gamma",
    "max_sub_iteration",
    "flux_threshold",
    "stop_rms_factor",
    "stop_peak_factor",
    "stop_cycle_factor",
    "stop_sidelobe_level",
    "max_iteration",
    "divergence_factor",
    "stall_threshold",
    "auto_mask",
    "force_auto_mask",
    "auto_mask_peak_threshold",
    "auto_mask_rms_threshold",
)


_DUMP_CTOR_ARGS = (
    "raw_psfs",
    "xdes",
    "scale_mask",
    "scale_sigmas",
    "scale_bias",
    "map_pixel_facet",
    "fft_padding",
)


def _save(directory, **arrays):
    os.makedirs(directory, exist_ok=True)
    for name, value in arrays.items():
        # None is an unset optional; NaN is how the replay reads it back.
        np.save(os.path.join(directory, name + ".npy"), np.nan if value is None else value)


class _DumpingDDMSC(ddmsc.DDMSC):
    """DDMSC that mirrors its inputs to $FAST_DECONV_DUMP for replay_ddmsc.

    Keyword arguments only: the dump reads the ctor inputs off the kwargs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dump_dir = os.environ["FAST_DECONV_DUMP"]
        self._dump_cycle = 0
        _save(
            os.path.join(self._dump_dir, "init"),
            **{k: kwargs[k] for k in _DUMP_CTOR_ARGS},
        )

    def run(self, dirty, jones_norm, weights_freq):
        # dirty is in/out, so it must be dumped before the call.
        _save(
            os.path.join(self._dump_dir, f"cycle_{self._dump_cycle}"),
            dirty=dirty,
            jones_norm=jones_norm,
            weights_freq=weights_freq,
            **{p: getattr(self, p) for p in _DUMP_PARAMS},
        )
        self._dump_cycle += 1
        return super().run(dirty, jones_norm, weights_freq)


DDMSC = _DumpingDDMSC if os.environ.get("FAST_DECONV_DUMP") else ddmsc.DDMSC

__all__ = ["ConvergenceStatus", "DDMSC", "DDMSCResult", "ddmsc"]
