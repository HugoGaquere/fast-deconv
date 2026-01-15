"""
Benchmark cases for fast-deconv operations.

Each case defines a comparison between fast-deconv and CuPy implementations.
"""

from .base import BenchmarkCase, BenchmarkConfig
from .argmax import ArgmaxBenchmark, ArgmaxAbsBenchmark, MaskedArgmaxBenchmark
from .subtract import SubtractBenchmark, SubtractStridedBenchmark
from .wscms import SubtractPsfFromDirtyBenchmark

__all__ = [
    "BenchmarkCase",
    "BenchmarkConfig",
    "ArgmaxBenchmark",
    "ArgmaxAbsBenchmark",
    "MaskedArgmaxBenchmark",
    "SubtractBenchmark",
    "SubtractStridedBenchmark",
    "SubtractPsfFromDirtyBenchmark",
]
