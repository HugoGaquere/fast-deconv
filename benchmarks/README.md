# Benchmark Framework

A modular benchmarking framework for comparing fast-deconv against CuPy.

## Quick Start

```bash
# Run all benchmarks
python -m benchmarks.runner

# Quick sanity check
python -m benchmarks.runner --quick

# Run specific category
python -m benchmarks.runner -c argmax
python -m benchmarks.runner -c subtract
python -m benchmarks.runner -c wscms

# Save results to JSON
python -m benchmarks.runner -o results.json

# List available benchmarks
python -m benchmarks.runner --list

# Validate correctness (no timing)
python -m benchmarks.runner --validate
```

## Benchmark Modes

| Mode | Warmup | Iterations | Use Case |
|------|--------|------------|----------|
| Default | 50 | 200 | Normal benchmarking |
| `--quick` | 10 | 50 | Quick sanity checks |
| `--thorough` | 100 | 500 | Statistical significance |
| `--profile` | 5 | 10 | Nsight profiling |

## Profiling with NVIDIA Nsight

### Nsight Systems (Timeline Analysis)

```bash
# Profile all operations
nsys profile -o profile_timeline python -m benchmarks.profile

# Profile specific operation
nsys profile -o argmax_timeline python -m benchmarks.profile -c argmax

# View results
nsys-ui profile_timeline.nsys-rep
```

### Nsight Compute (Kernel Analysis)

```bash
# Full kernel analysis
ncu --set full -o kernel_analysis python -m benchmarks.profile -c subtract

# Roofline analysis
ncu --set roofline -o roofline python -m benchmarks.profile

# Profile specific kernel (skip warmup)
ncu --kernel-name "subtract" --launch-skip 5 -o subtract_kernel python -m benchmarks.profile -c subtract

# View results
ncu-ui kernel_analysis.ncu-rep
```

### Useful Nsight Compute Options

```bash
--set full          # Comprehensive metrics
--set roofline      # Roofline model analysis
--set memory        # Memory throughput focus
--kernel-name <re>  # Filter by kernel name (regex)
--launch-skip N     # Skip first N launches (warmup)
--launch-count N    # Profile only N launches
```

## Adding New Benchmarks

### 1. Create a new benchmark case

```python
# benchmarks/cases/my_operation.py
from dataclasses import dataclass, field
from typing import Any, Callable
import cupy as cp
import _fast_deconv as fd
from .base import BenchmarkCase

@dataclass
class MyOperationBenchmark(BenchmarkCase):
    """Benchmark description."""

    # Configuration parameters
    size: int = 1024

    # Required fields (set in __post_init__)
    name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self):
        self.name = f"my_operation_{self.size}"
        self.description = f"My operation with size {self.size}"

    def setup(self) -> tuple[Callable[[], Any], Callable[[], Any]]:
        # Setup data
        data = cp.random.randn(self.size, dtype=cp.float32)
        resources = fd.stream_resources()

        def cupy_fn():
            # CuPy equivalent implementation
            return cp.some_operation(data)

        def fast_deconv_fn():
            # fast-deconv implementation
            return fd.my_operation(data, resources)

        return cupy_fn, fast_deconv_fn

    def get_metadata(self) -> dict[str, Any]:
        return {"size": self.size}
```

### 2. Register in `cases/__init__.py`

```python
from .my_operation import MyOperationBenchmark
__all__ = [..., "MyOperationBenchmark"]
```

### 3. Add to runner registry

```python
# benchmarks/runner.py
BENCHMARK_REGISTRY = {
    ...
    "my_category": [
        MyOperationBenchmark(size=1024),
        MyOperationBenchmark(size=4096),
    ],
}
```

### 4. Add profiling case (optional)

```python
# benchmarks/profile.py
def profile_my_operation():
    """Profile my operation."""
    # Setup and warmup
    ...
    # Profiled iterations with NVTX markers
    for i in range(10):
        with cp.cuda.nvtx.Mark(f"my_operation_iter_{i}"):
            fd.my_operation(...)

PROFILE_CASES["my_operation"] = profile_my_operation
```

## Architecture

```
benchmarks/
├── __init__.py
├── core.py              # Core timing utilities
├── runner.py            # CLI benchmark runner
├── profile.py           # Nsight profiling script
├── README.md
└── cases/
    ├── __init__.py
    ├── base.py          # BenchmarkCase base class
    ├── argmax.py        # Argmax benchmarks
    ├── subtract.py      # Subtract benchmarks
    └── wscms.py         # WSCMS algorithm benchmarks
```

## Key Features

- **CUDA Event Timing**: Uses GPU events for accurate timing (not wall-clock)
- **Warmup Phase**: Handles JIT compilation and driver initialization
- **Statistical Analysis**: Reports median, mean, std, percentiles
- **NVTX Markers**: Operations are labeled for easy identification in Nsight
- **Validation**: Can verify correctness before timing
- **Modular Design**: Easy to add new operations and configurations

## Output Example

```
================================================================================
BENCHMARK SUMMARY
================================================================================

Comparison: fast_deconv:argmax_2d vs cupy:argmax_2d
  Baseline:  cupy:argmax_2d: median=0.1234ms, mean=0.1256ms (±0.0045), [p5=0.1189, p95=0.1345]
  Contender: fast_deconv:argmax_2d: median=0.0567ms, mean=0.0578ms (±0.0023), [p5=0.0543, p95=0.0612]
  Speedup: 2.18x (FASTER, +54.1%)

================================================================================
Total benchmarks: 25
Faster than baseline: 23/25
Geometric mean speedup: 1.87x
================================================================================
```
