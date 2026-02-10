"""
Benchmark cases for fast-deconv operations.

Each case defines a comparison between fast-deconv and CuPy implementations.
"""

from .base import BenchmarkCase, BenchmarkConfig
from .wscms import WscmsMinorCycleBenchmark

__all__ = [
    "BenchmarkCase",
    "BenchmarkConfig",
    "WscmsMinorCycleBenchmark",
]
