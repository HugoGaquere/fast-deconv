"""Shared plumbing for the paper-data harness.

Holds the run context passed to every stage, subprocess execution with
tee-to-logfile, and small CSV/JSON helpers. Stages are modules in stages/
exposing run(ctx) -> dict (the stage summary, persisted to summary.json).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


class SkipStage(Exception):
    """Raised by a stage when its prerequisites are absent (not an error)."""


@dataclass
class Ctx:
    repo_root: Path
    bin_dir: Path
    bundle: Path
    device: int = 0
    preset_name: str = "full"
    preset: dict = field(default_factory=dict)
    dump_dir: str = ""
    ref_dir: str = ""
    cycles: str = "1"
    label: str = ""
    dump_gpu_output: bool = False
    force_auto_mask_last: bool = False
    env: dict = field(default_factory=dict)

    def binary(self, name: str) -> Path:
        p = self.bin_dir / name
        if not p.exists():
            raise RuntimeError(
                f"{p} not found -- build the project first (see scripts/paper/README.md)")
        return p

    def stage_dir(self, name: str) -> Path:
        d = self.bundle / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def run_cmd(ctx: Ctx, args, log_name: str, check: bool = True) -> int:
    """Run a command, echoing output and teeing it to logs/<log_name>.log."""
    logdir = ctx.bundle / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / f"{log_name}.log"
    args = [str(a) for a in args]
    print(f"  $ {' '.join(args)}")
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(args) + "\n\n")
        log.flush()
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write("    " + line)
            log.write(line)
        proc.wait()
    if check and proc.returncode != 0:
        raise RuntimeError(f"{args[0]} exited with {proc.returncode} (see {log_path})")
    return proc.returncode


def capture_cmd(args) -> str:
    """Run a command and return stdout; raises on non-zero exit."""
    res = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{args[0]} failed: {res.stderr.strip()[:500]}")
    return res.stdout


def try_capture(args) -> str | None:
    """capture_cmd that returns None instead of raising (for optional tools)."""
    try:
        return capture_cmd(args)
    except (RuntimeError, FileNotFoundError, OSError):
        return None


def write_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + "\n")


def read_json(path) -> dict:
    return json.loads(Path(path).read_text())


def read_csv_rows(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def kernel_ident(name: str) -> str:
    """Unqualified kernel identifier -- the only spelling ncu and nsys agree on
    (they demangle namespaces, casts, and template args differently). Also
    merges template instantiations and overloads of the same kernel."""
    head = name.split("<")[0].split("(")[0].removeprefix("void ").strip()
    return head.split("::")[-1] or name.strip()


def first_cycle(spec: str) -> int:
    """First cycle id in a spec like '1,2,4-6' or '3-7' (-> 1, 3 respectively)."""
    return int(spec.split(",")[0].split("-")[0].strip())


def iteration_flags(cfg: dict) -> list[str]:
    """example_ddmsc --max-clean-iter/--max-iter flags from a stage cfg. Omitted
    when the cfg does not set them, so the dump's full schedule runs."""
    flags = []
    if cfg.get("max_clean_iter"):
        flags.append(f"--max-clean-iter={cfg['max_clean_iter']}")
    if cfg.get("max_iter"):
        flags.append(f"--max-iter={cfg['max_iter']}")
    return flags


def to_num(s):
    """Best-effort numeric conversion ('1,234.5' -> float); strings pass through."""
    if not isinstance(s, str):
        return s
    t = s.strip().replace(",", "")
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return s.strip()
