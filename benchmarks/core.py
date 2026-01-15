"""
Core benchmarking utilities with CUDA event timing.

This module provides the foundation for GPU benchmarking with:
- CUDA event-based timing (more accurate than wall-clock)
- Warm-up phases for JIT/driver initialization
- Statistical analysis of results
- NVTX markers for Nsight profiling integration
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any
import cupy as cp
import numpy as np

import nvtx

# Re-export nvtx.annotate as nvtx_range for convenience
nvtx_range = nvtx.annotate


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    times_ms: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def median(self) -> float:
        return float(np.median(self.times_ms))

    @property
    def mean(self) -> float:
        return float(np.mean(self.times_ms))

    @property
    def std(self) -> float:
        return float(np.std(self.times_ms))

    @property
    def min(self) -> float:
        return float(np.min(self.times_ms))

    @property
    def max(self) -> float:
        return float(np.max(self.times_ms))

    @property
    def p5(self) -> float:
        return float(np.percentile(self.times_ms, 5))

    @property
    def p95(self) -> float:
        return float(np.percentile(self.times_ms, 95))

    @property
    def iterations(self) -> int:
        return len(self.times_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "median_ms": self.median,
            "mean_ms": self.mean,
            "std_ms": self.std,
            "min_ms": self.min,
            "max_ms": self.max,
            "p5_ms": self.p5,
            "p95_ms": self.p95,
            "iterations": self.iterations,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        return (
            f"{self.name}: median={self.median:.4f}ms, "
            f"mean={self.mean:.4f}ms (±{self.std:.4f}), "
            f"[p5={self.p5:.4f}, p95={self.p95:.4f}]"
        )


@dataclass
class ComparisonResult:
    """Results comparing two implementations."""

    baseline: BenchmarkResult
    contender: BenchmarkResult

    @property
    def speedup(self) -> float:
        """Speedup factor (>1 means contender is faster)."""
        return self.baseline.median / self.contender.median

    @property
    def improvement_pct(self) -> float:
        """Percentage improvement (positive means contender is faster)."""
        return (1 - self.contender.median / self.baseline.median) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "contender": self.contender.to_dict(),
            "speedup": self.speedup,
            "improvement_pct": self.improvement_pct,
        }

    def __str__(self) -> str:
        status = "FASTER" if self.speedup > 1 else "SLOWER"
        return (
            f"Comparison: {self.contender.name} vs {self.baseline.name}\n"
            f"  Baseline:  {self.baseline}\n"
            f"  Contender: {self.contender}\n"
            f"  Speedup: {self.speedup:.2f}x ({status}, {self.improvement_pct:+.1f}%)"
        )


def benchmark_gpu(
    fn: Callable[[], Any],
    name: str = "benchmark",
    warmup: int = 50,
    iterations: int = 200,
    use_nvtx: bool = True,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """
    Benchmark a GPU function using CUDA events.

    Args:
        fn: Function to benchmark (should perform GPU operations)
        name: Name for this benchmark
        warmup: Number of warm-up iterations (not timed)
        iterations: Number of timed iterations
        use_nvtx: Whether to wrap execution in NVTX markers
        metadata: Optional metadata to attach to results

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warm-up phase
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()

    # Timed runs with CUDA events
    times: list[float] = []

    for _ in range(iterations):
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        if use_nvtx:
            with nvtx_range(name):
                fn()
        else:
            fn()
        end.record()

        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))

    return BenchmarkResult(
        name=name,
        times_ms=times,
        metadata=metadata or {},
    )


def compare(
    baseline_fn: Callable[[], Any],
    contender_fn: Callable[[], Any],
    baseline_name: str = "baseline",
    contender_name: str = "contender",
    warmup: int = 50,
    iterations: int = 200,
    use_nvtx: bool = True,
    metadata: dict[str, Any] | None = None,
) -> ComparisonResult:
    """
    Compare two GPU implementations.

    Args:
        baseline_fn: Baseline implementation (e.g., CuPy)
        contender_fn: Contender implementation (e.g., fast-deconv)
        baseline_name: Name for baseline
        contender_name: Name for contender
        warmup: Number of warm-up iterations
        iterations: Number of timed iterations
        use_nvtx: Whether to use NVTX markers
        metadata: Optional metadata for both results

    Returns:
        ComparisonResult with speedup analysis
    """
    baseline = benchmark_gpu(
        baseline_fn,
        name=baseline_name,
        warmup=warmup,
        iterations=iterations,
        use_nvtx=use_nvtx,
        metadata=metadata,
    )

    contender = benchmark_gpu(
        contender_fn,
        name=contender_name,
        warmup=warmup,
        iterations=iterations,
        use_nvtx=use_nvtx,
        metadata=metadata,
    )

    return ComparisonResult(baseline=baseline, contender=contender)


def save_results(
    results: list[ComparisonResult] | list[BenchmarkResult],
    path: str | Path,
) -> None:
    """Save benchmark results to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.to_dict() for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def print_summary(results: list[ComparisonResult]) -> None:
    """Print a summary table of comparison results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    for result in results:
        print(f"\n{result}")

    print("\n" + "=" * 80)

    # Summary statistics
    speedups = [r.speedup for r in results]
    faster_count = sum(1 for s in speedups if s > 1)
    print(f"Total benchmarks: {len(results)}")
    print(f"Faster than baseline: {faster_count}/{len(results)}")
    print(f"Geometric mean speedup: {np.exp(np.mean(np.log(speedups))):.2f}x")
    print("=" * 80 + "\n")
