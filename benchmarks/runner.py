"""
Benchmark runner with CLI interface.

Usage:
    # Run all benchmarks
    python -m benchmarks.runner

    # Run specific categories
    python -m benchmarks.runner --category argmax
    python -m benchmarks.runner --category subtract
    python -m benchmarks.runner --category wscms

    # Quick mode for sanity checks
    python -m benchmarks.runner --quick

    # Include large-scale benchmarks (10000x10000+)
    python -m benchmarks.runner --large

    # Profiling mode (few iterations for Nsight)
    python -m benchmarks.runner --profile

    # Save results to JSON
    python -m benchmarks.runner --output results.json

    # Validate correctness only (no timing)
    python -m benchmarks.runner --validate
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from benchmarks.core import ComparisonResult, print_summary, save_results
from benchmarks.cases import (
    BenchmarkCase,
    BenchmarkConfig,
    ArgmaxBenchmark,
    ArgmaxAbsBenchmark,
    MaskedArgmaxBenchmark,
    SubtractBenchmark,
    SubtractStridedBenchmark,
    SubtractPsfFromDirtyBenchmark,
    CleanDirtiesBenchmark,
    CleanDirtiesStridedBenchmark,
)
from benchmarks.cases.subtract import SubtractInPlaceBenchmark
from benchmarks.cases.wscms import SubtractPsfFromDirtyStridedBenchmark


# Registry of all benchmark cases organized by category
BENCHMARK_REGISTRY: dict[str, list[BenchmarkCase]] = {
    "argmax": [
        # Different sizes
        ArgmaxBenchmark(shape=(1024,)),
        ArgmaxBenchmark(shape=(1024, 1024)),
        ArgmaxBenchmark(shape=(256, 256, 256)),
        ArgmaxBenchmark(shape=(64, 64, 64, 64)),
        # Absolute value variants
        ArgmaxAbsBenchmark(shape=(1024, 1024)),
        ArgmaxAbsBenchmark(shape=(256, 256, 256)),
        # Masked variants
        MaskedArgmaxBenchmark(shape=(1024, 1024), mask_ratio=0.3),
        MaskedArgmaxBenchmark(shape=(1024, 1024), mask_ratio=0.7),
        MaskedArgmaxBenchmark(shape=(256, 256, 256), mask_ratio=0.5),
    ],
    "subtract": [
        # Contiguous arrays - various sizes
        SubtractBenchmark(shape=(1024,)),
        SubtractBenchmark(shape=(1024, 1024)),
        SubtractBenchmark(shape=(2048, 2048)),
        SubtractBenchmark(shape=(256, 256, 256)),
        SubtractBenchmark(shape=(64, 64, 64, 64)),
        # Strided arrays
        SubtractStridedBenchmark(shape=(512, 512), stride_factor=2),
        SubtractStridedBenchmark(shape=(1024, 1024), stride_factor=2),
        SubtractStridedBenchmark(shape=(128, 128, 128), stride_factor=2),
        # In-place operations
        SubtractInPlaceBenchmark(shape=(1024, 1024)),
        SubtractInPlaceBenchmark(shape=(2048, 2048)),
    ],
    "wscms": [
        # Standard image sizes
        SubtractPsfFromDirtyBenchmark(n_channels=8, height=256, width=256),
        SubtractPsfFromDirtyBenchmark(n_channels=16, height=512, width=512),
        SubtractPsfFromDirtyBenchmark(n_channels=32, height=512, width=512),
        SubtractPsfFromDirtyBenchmark(n_channels=16, height=1024, width=1024),
        # Strided variants (realistic memory layouts)
        SubtractPsfFromDirtyStridedBenchmark(n_channels=16, height=512, width=512),
        SubtractPsfFromDirtyStridedBenchmark(n_channels=32, height=512, width=512),
        # Fused clean_dirties operation
        CleanDirtiesBenchmark(n_channels=8, height=256, width=256),
        CleanDirtiesBenchmark(n_channels=16, height=512, width=512),
        CleanDirtiesBenchmark(n_channels=32, height=512, width=512),
        CleanDirtiesBenchmark(n_channels=16, height=1024, width=1024),
        # Strided clean_dirties
        CleanDirtiesStridedBenchmark(n_channels=16, height=512, width=512),
        CleanDirtiesStridedBenchmark(n_channels=32, height=512, width=512),
    ],
}

# Large-scale benchmarks (10000x10000 and bigger) - only run with --large flag
LARGE_BENCHMARK_REGISTRY: dict[str, list[BenchmarkCase]] = {
    "wscms": [
        # Large-scale clean_dirties (radio astronomy scale)
        CleanDirtiesBenchmark(n_channels=16, height=10000, width=10000),
        CleanDirtiesBenchmark(n_channels=4, height=20000, width=20000),
    ],
}


def get_benchmarks(
    categories: Sequence[str] | None = None,
    include_large: bool = False,
) -> list[BenchmarkCase]:
    """Get benchmark cases for specified categories (or all if None).

    Args:
        categories: List of categories to include, or None for all.
        include_large: If True, include large-scale benchmarks (10000x10000+).
    """
    if categories is None:
        categories = list(BENCHMARK_REGISTRY.keys())

    benchmarks = []
    for cat in categories:
        if cat not in BENCHMARK_REGISTRY:
            print(f"Warning: Unknown category '{cat}', skipping")
            continue
        benchmarks.extend(BENCHMARK_REGISTRY[cat])

        # Add large benchmarks if requested
        if include_large and cat in LARGE_BENCHMARK_REGISTRY:
            benchmarks.extend(LARGE_BENCHMARK_REGISTRY[cat])

    return benchmarks


def validate_all(benchmarks: list[BenchmarkCase]) -> bool:
    """Validate all benchmarks produce correct results."""
    print("Validating benchmark correctness...")
    all_valid = True

    for bench in benchmarks:
        try:
            valid = bench.validate()
            status = "OK" if valid else "FAIL"
            print(f"  [{status}] {bench.name}")
            if not valid:
                all_valid = False
        except Exception as e:
            print(f"  [ERROR] {bench.name}: {e}")
            all_valid = False

    return all_valid


def run_benchmarks(
    benchmarks: list[BenchmarkCase],
    config: BenchmarkConfig,
) -> list[ComparisonResult]:
    """Run all benchmarks and return results."""
    results = []

    print(f"\nRunning {len(benchmarks)} benchmarks...")
    print(f"Config: warmup={config.warmup}, iterations={config.iterations}")
    print("-" * 60)

    for i, bench in enumerate(benchmarks, 1):
        print(f"[{i}/{len(benchmarks)}] {bench.name}...", end=" ", flush=True)
        try:
            result = bench.run(config)
            results.append(result)
            print(f"done ({result.speedup:.2f}x)")
        except Exception as e:
            print(f"ERROR: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fast-deconv against CuPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--category",
        "-c",
        action="append",
        choices=list(BENCHMARK_REGISTRY.keys()),
        help="Run only specific category (can be repeated)",
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Quick mode (fewer iterations)",
    )
    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        help="Profiling mode (minimal iterations for Nsight)",
    )
    parser.add_argument(
        "--thorough",
        "-t",
        action="store_true",
        help="Thorough mode (more iterations for statistical significance)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--validate",
        "-v",
        action="store_true",
        help="Only validate correctness, no timing",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        help="Override warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        help="Override number of iterations",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available benchmarks and exit",
    )
    parser.add_argument(
        "--large",
        "-L",
        action="store_true",
        help="Include large-scale benchmarks (10000x10000 and bigger)",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("Available benchmarks:")
        for cat, benchmarks in BENCHMARK_REGISTRY.items():
            print(f"\n{cat}:")
            for bench in benchmarks:
                print(f"  - {bench.name}: {bench.description}")
        if LARGE_BENCHMARK_REGISTRY:
            print("\nLarge-scale benchmarks (--large flag):")
            for cat, benchmarks in LARGE_BENCHMARK_REGISTRY.items():
                print(f"\n{cat}:")
                for bench in benchmarks:
                    print(f"  - {bench.name}: {bench.description}")
        return 0

    # Get benchmarks
    benchmarks = get_benchmarks(args.category, include_large=args.large)
    if not benchmarks:
        print("No benchmarks to run")
        return 1

    # Validate mode
    if args.validate:
        success = validate_all(benchmarks)
        return 0 if success else 1

    # Determine config
    if args.profile:
        config = BenchmarkConfig.for_profiling()
    elif args.quick:
        config = BenchmarkConfig.quick()
    elif args.thorough:
        config = BenchmarkConfig.thorough()
    else:
        config = BenchmarkConfig()

    # Override with explicit values
    if args.warmup is not None:
        config.warmup = args.warmup
    if args.iterations is not None:
        config.iterations = args.iterations

    # Run benchmarks
    results = run_benchmarks(benchmarks, config)

    # Print summary
    if results:
        print_summary(results)

    # Save results
    if args.output and results:
        save_results(results, args.output)
        print(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
