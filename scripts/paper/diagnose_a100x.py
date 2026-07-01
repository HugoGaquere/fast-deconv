#!/usr/bin/env python3
"""Diagnose why the A100X delivers less bandwidth than its datasheet peak.

Self-contained. Run it ON the host that has the GPU, then paste the whole
output back. It executes the live diagnostics and an on-device HBM bandwidth
probe, sampling clocks / throttle reasons *during* the probe to catch a droop.

Examples
--------
    # GPU is nvidia-smi index 4 on this host:
    python scripts/paper/diagnose_a100x.py --device 4

    # If you instead pin the card via the environment, index 0 is that card:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
        python scripts/paper/diagnose_a100x.py

The on-device bandwidth probe uses cupy if available (the repo venv has
cupy-cuda12x). Without cupy it falls back to nvidia-smi-only diagnostics.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def sh(args, timeout=60):
    """Run a command, return (rc, stdout+stderr). Never raises."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except FileNotFoundError:
        return 127, f"[not found: {args[0]}]"
    except subprocess.TimeoutExpired:
        return 124, f"[timed out after {timeout}s: {' '.join(map(str, args))}]"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def smi(device, query, extra=None):
    args = ["nvidia-smi", "-i", str(device),
            f"--query-gpu={query}", "--format=csv"]
    if extra:
        args[1:1] = extra
    rc, out = sh(args)
    return out.strip()


# --------------------------------------------------------------------------- #
#  clock / throttle sampler (runs in a thread during the bandwidth probe)
# --------------------------------------------------------------------------- #

class Sampler(threading.Thread):
    Q = ("clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,"
         "clocks_throttle_reasons.active,utilization.gpu")

    def __init__(self, device, period=0.5):
        super().__init__(daemon=True)
        self.device, self.period = device, period
        self.rows, self._stop_evt = [], threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            rc, out = sh(["nvidia-smi", "-i", str(self.device),
                          f"--query-gpu={self.Q}",
                          "--format=csv,noheader,nounits"])
            if rc == 0:
                self.rows.append(out.strip())
            self._stop_evt.wait(self.period)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)


# --------------------------------------------------------------------------- #
#  on-device HBM bandwidth probe (device-to-device memcpy)
# --------------------------------------------------------------------------- #

def _busid(props):
    """nvidia-smi-style PCI bus id 'DOMAIN:BUS:DEV.0' from cupy device props."""
    return (f"{props['pciDomainID']:08X}:{props['pciBusID']:02X}:"
            f"{props['pciDeviceID']:02X}.0")


def hbm_bandwidth(cuda_index, seconds=5.0, buf_gib=2.0, expected_busid=None):
    """Sustained D2D copy bandwidth in GB/s, or a message if cupy is absent.

    Verifies the CUDA-selected device is the one nvidia-smi addressed: CUDA's
    default ordering differs from nvidia-smi's, so the indices need not match."""
    try:
        import cupy as cp
    except Exception as e:  # noqa: BLE001
        return None, f"[cupy unavailable: {e}; skipping on-device probe]"

    cp.cuda.Device(cuda_index).use()
    props = cp.cuda.runtime.getDeviceProperties(cuda_index)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    busid = _busid(props)
    if expected_busid and busid.upper() != expected_busid.upper():
        return None, (f"[DEVICE MISMATCH] cupy index {cuda_index} = {name} @ {busid}, "
                      f"but nvidia-smi target is @ {expected_busid}. "
                      f"Set CUDA_DEVICE_ORDER=PCI_BUS_ID (the script does this) or "
                      f"pick the cupy index whose bus id matches.")

    nbytes = int(buf_gib * 1024**3) & ~0xFFFF
    try:
        a = cp.empty(nbytes, dtype=cp.uint8)
        b = cp.empty(nbytes, dtype=cp.uint8)
    except Exception as e:  # noqa: BLE001
        return None, f"[alloc {buf_gib} GiB x2 failed: {e}]"

    D2D = cp.cuda.runtime.memcpyDeviceToDevice
    for _ in range(5):  # warmup
        cp.cuda.runtime.memcpy(b.data.ptr, a.data.ptr, nbytes, D2D)
    cp.cuda.Stream.null.synchronize()

    start, end = cp.cuda.Event(), cp.cuda.Event()
    reps, t0 = 0, time.time()
    start.record()
    while time.time() - t0 < seconds:
        cp.cuda.runtime.memcpy(b.data.ptr, a.data.ptr, nbytes, D2D)
        reps += 1
    end.record()
    end.synchronize()
    ms = cp.cuda.get_elapsed_time(start, end)
    moved = reps * nbytes * 2  # read + write
    gbps = moved / (ms / 1e3) / 1e9
    del a, b
    cp.get_default_memory_pool().free_all_blocks()
    return gbps, (f"cupy device {cuda_index} = {name} @ {busid}; "
                  f"{reps} copies of {buf_gib} GiB")


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #

def resolve_indices(arg_device):
    """Return (smi_index, cuda_index). If CUDA_VISIBLE_DEVICES pins one integer
    card, nvidia-smi must address that physical index while cupy sees it as 0."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if arg_device is not None:
        return arg_device, (0 if cvd.isdigit() and int(cvd) == arg_device else arg_device)
    if cvd.isdigit():
        return int(cvd), 0
    return 0, 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=None,
                    help="nvidia-smi GPU index (default: from CUDA_VISIBLE_DEVICES, else 0)")
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="duration of the sustained bandwidth probe")
    ap.add_argument("--buf-gib", type=float, default=2.0,
                    help="size of each D2D buffer in GiB")
    ap.add_argument("--no-bandwidth", action="store_true",
                    help="skip the on-device bandwidth probe (nvidia-smi only)")
    args = ap.parse_args()

    # Make CUDA enumerate by PCI bus id so cupy's ordinal matches nvidia-smi's.
    # Must be set before the CUDA runtime initializes (i.e. before cupy import).
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    smi_idx, cuda_idx = resolve_indices(args.device)

    print("#" * 70)
    print("# A100X DIAGNOSTIC -- paste everything below this line back to Claude")
    print("#" * 70)
    print(f"# host={os.uname().nodename}  smi_index={smi_idx}  cuda_index={cuda_idx}"
          f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')}"
          f"  CUDA_DEVICE_ORDER=PCI_BUS_ID"
          f"  time={time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not shutil.which("nvidia-smi"):
        print("\n[fatal] nvidia-smi not on PATH -- run this on the GPU host.")
        return 1

    section("0. Identity (confirm this is the A100X)")
    print(smi(smi_idx, "name,uuid,driver_version,vbios_version,compute_cap"))

    section("1. PCIe link actually negotiated (converged/network boards downshift)")
    print(smi(smi_idx, "pcie.link.gen.current,pcie.link.gen.max,"
                       "pcie.link.width.current,pcie.link.width.max"))
    print("\nidle vs max (current may be low when idle -- compare gen/width maxes):")
    rc, out = sh(["nvidia-smi", "-i", str(smi_idx), "-q", "-d", "PCI"])
    for ln in out.splitlines():
        if any(k in ln for k in ("Link Width", "Link Gen", "Bus Id", "Device Id",
                                 "Max", "Current")):
            print("   " + ln.strip())

    section("2. Clocks: configured vs max (is mem clock pinned below its ceiling?)")
    print(smi(smi_idx, "clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,"
                       "clocks.applications.gr,clocks.applications.mem"))

    section("3. Throttle reasons (idle snapshot)")
    rc, out = sh(["nvidia-smi", "-i", str(smi_idx), "-q", "-d", "PERFORMANCE"])
    grab = False
    for ln in out.splitlines():
        if "Clocks Event Reasons" in ln or "Clocks Throttle Reasons" in ln:
            grab = True
        if grab and ln.strip():
            print("   " + ln.rstrip())
        if grab and not ln.strip():
            grab = False

    section("4. ECC, MIG, persistence, power cap (each can cap delivered BW)")
    print(smi(smi_idx, "ecc.mode.current,ecc.mode.pending,mig.mode.current,"
                       "persistence_mode,power.draw,power.limit,"
                       "enforced.power.limit,temperature.gpu"))

    if not args.no_bandwidth:
        section("5. ON-DEVICE HBM BANDWIDTH (decisive: vs datasheet 2039 GB/s)")
        target_busid = smi(smi_idx, "pci.bus_id").splitlines()[-1].strip()
        sampler = Sampler(smi_idx)
        sampler.start()
        try:
            gbps, note = hbm_bandwidth(cuda_idx, args.seconds, args.buf_gib,
                                       expected_busid=target_busid)
        finally:
            try:
                sampler.stop()
            except Exception as e:  # noqa: BLE001
                print(f"[sampler stop warning: {e}]")
        print(note)
        if gbps is not None:
            print(f"\n  >>> measured D2D HBM bandwidth: {gbps:,.0f} GB/s <<<")
            print(f"      ({100*gbps/2039:.1f}% of A100X datasheet peak 2039 GB/s)")
        print("\n  clocks/throttle sampled DURING the probe "
              "(sm, mem, pwr.draw, pwr.lim, temp, throttle_bitmask, util):")
        for r in sampler.rows[:20]:
            print("   " + r)
        if not sampler.rows:
            print("   [no samples captured]")

    print("\n" + "#" * 70)
    print("# END -- paste from the first '#' line through this one")
    print("#" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
