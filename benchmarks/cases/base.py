"""
Base classes for benchmark cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import cupy as cp

from benchmarks.core import ComparisonResult, compare


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    warmup: int = 50
    iterations: int = 200
    use_nvtx: bool = True

    # For Nsight profiling (fewer iterations)
    @classmethod
    def for_profiling(cls) -> BenchmarkConfig:
        """Config optimized for Nsight profiling."""
        return cls(warmup=5, iterations=10, use_nvtx=True)

    # For quick sanity checks
    @classmethod
    def quick(cls) -> BenchmarkConfig:
        """Config for quick sanity checks."""
        return cls(warmup=10, iterations=50, use_nvtx=False)

    # For thorough benchmarking
    @classmethod
    def thorough(cls) -> BenchmarkConfig:
        """Config for thorough statistical analysis."""
        return cls(warmup=100, iterations=500, use_nvtx=True)


@dataclass
class BenchmarkCase(ABC):
    """
    Base class for benchmark cases.

    Subclasses must implement:
    - setup(): Prepare data and return (cupy_fn, fast_deconv_fn)
    - name: A descriptive name for the benchmark
    - description: What this benchmark tests

    Usage:
        case = MyBenchmarkCase(shape=(1000, 1000))
        result = case.run()
        print(result)
    """

    # Override these in subclasses
    name: str = field(init=False)
    description: str = field(init=False)

    @abstractmethod
    def setup(self) -> tuple[Callable[[], Any], Callable[[], Any]]:
        """
        Set up the benchmark and return callables.

        Returns:
            Tuple of (cupy_fn, fast_deconv_fn) where each is a
            zero-argument callable that performs the operation.
        """
        ...

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata about this benchmark configuration."""
        return {}

    def run(self, config: BenchmarkConfig | None = None) -> ComparisonResult:
        """
        Run the benchmark comparison.

        Args:
            config: Benchmark configuration (uses default if None)

        Returns:
            ComparisonResult with timing data
        """
        if config is None:
            config = BenchmarkConfig()

        cupy_fn, fast_deconv_fn = self.setup()

        return compare(
            baseline_fn=cupy_fn,
            contender_fn=fast_deconv_fn,
            baseline_name=f"cupy:{self.name}",
            contender_name=f"fast_deconv:{self.name}",
            warmup=config.warmup,
            iterations=config.iterations,
            use_nvtx=config.use_nvtx,
            metadata=self.get_metadata(),
        )

    def validate(self) -> bool:
        """
        Validate that both implementations produce the same result.

        Returns:
            True if outputs match, False otherwise
        """
        cupy_fn, fast_deconv_fn = self.setup()

        # Run both and compare
        cp.cuda.Stream.null.synchronize()
        cupy_result = cupy_fn()
        cp.cuda.Stream.null.synchronize()
        fast_deconv_result = fast_deconv_fn()
        cp.cuda.Stream.null.synchronize()

        return self._compare_results(cupy_result, fast_deconv_result)

    def _compare_results(self, cupy_result: Any, fast_deconv_result: Any) -> bool:
        """Compare results from both implementations. Override for custom comparison."""
        if isinstance(cupy_result, cp.ndarray) and isinstance(
            fast_deconv_result, cp.ndarray
        ):
            return cp.allclose(cupy_result, fast_deconv_result, rtol=1e-5, atol=1e-5)
        elif isinstance(cupy_result, tuple) and isinstance(fast_deconv_result, tuple):
            # For argmax-like returns (index, value)
            return all(
                self._compare_results(a, b)
                for a, b in zip(cupy_result, fast_deconv_result)
            )
        else:
            return cupy_result == fast_deconv_result
