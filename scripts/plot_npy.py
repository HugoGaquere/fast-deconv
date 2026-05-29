#!/usr/bin/env python3
"""Plot a .npy dump produced by fast_deconv::util::dump_npy.

Examples:
    plot_npy.py residual.npy                           # show interactively
    plot_npy.py dirty.npy --index 3                    # pick freq channel 3 of (F,H,W)
    plot_npy.py psfs.npy --slice 0,2 --grid            # reduce leading dims, tile remainder
    plot_npy.py mask.npy --cmap gray
    plot_npy.py residual.npy --symlog --save out.png   # save single

    plot_npy.py *.npy                                  # batch: convert each to <stem>.png
    plot_npy.py                                        # batch all *.npy in CWD
    plot_npy.py *.npy --out-dir png/                   # write to png/ instead of CWD
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm


def parse_slice(spec: str, ndim: int) -> tuple[int, ...]:
    """Parse "0,2,1" into a tuple of leading-axis indices."""
    if not spec:
        return ()
    idx = tuple(int(x) for x in spec.split(","))
    if len(idx) >= ndim:
        sys.exit(f"--slice {spec} reduces all {ndim} dims; nothing left to plot")
    return idx


def reduce_leading(arr: np.ndarray, idx: tuple[int, ...]) -> np.ndarray:
    for i, k in enumerate(idx):
        if k < 0 or k >= arr.shape[0]:
            sys.exit(f"--slice index {k} out of range for axis (size {arr.shape[0]})")
        arr = arr[k]
    return arr


def make_norm(arr: np.ndarray, args) -> object | None:
    if args.log:
        positive = arr[arr > 0]
        if positive.size == 0:
            sys.exit("--log requires positive values; data has none")
        vmin = args.vmin if args.vmin is not None else float(positive.min())
        vmax = args.vmax if args.vmax is not None else float(positive.max())
        return LogNorm(vmin=vmin, vmax=vmax)
    if args.symlog:
        peak = float(np.nanmax(np.abs(arr)))
        linthresh = peak * 1e-3 if peak > 0 else 1e-6
        return SymLogNorm(linthresh=linthresh, vmin=args.vmin, vmax=args.vmax, base=10)
    return None


def plot_1d(arr: np.ndarray, ax, args):
    ax.plot(arr)
    ax.set_xlabel("index")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    if args.log:
        ax.set_yscale("log")
    elif args.symlog:
        ax.set_yscale("symlog")


def plot_2d(arr: np.ndarray, ax, args, fig):
    norm = make_norm(arr, args)
    kw = dict(cmap=args.cmap, origin="lower", interpolation="nearest")
    if norm is not None:
        kw["norm"] = norm
    else:
        if args.vmin is not None:
            kw["vmin"] = args.vmin
        if args.vmax is not None:
            kw["vmax"] = args.vmax
    im = ax.imshow(arr, **kw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_grid(arr: np.ndarray, args):
    """Tile leading-axis slices of a 3D array."""
    n = arr.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < n:
            plot_2d(arr[i], ax, args, fig)
            ax.set_title(f"[{i}]")
        else:
            ax.axis("off")
    return fig


def render(path: Path, args):
    """Load one .npy and build a Figure. Returns (fig, info_string) or (None, msg) on skip."""
    arr = np.load(path)
    info = f"{path}: shape={arr.shape} dtype={arr.dtype}"

    if args.abs:
        arr = np.abs(arr)
    if args.index is not None:
        arr = reduce_leading(arr, (args.index,))
    arr = reduce_leading(arr, parse_slice(args.slice, arr.ndim))

    finite = arr[np.isfinite(arr)]
    if finite.size:
        info += (f"  min={finite.min():.6g} max={finite.max():.6g} "
                 f"mean={finite.mean():.6g} std={finite.std():.6g}")

    title = args.title or f"{path.name}  shape={arr.shape}  dtype={arr.dtype}"

    if arr.ndim == 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_1d(arr, ax, args)
        ax.set_title(title)
    elif arr.ndim == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_2d(arr, ax, args, fig)
        ax.set_title(title)
    elif arr.ndim == 3 and args.grid:
        fig = plot_grid(arr, args)
        fig.suptitle(title)
        fig.tight_layout()
    elif arr.ndim == 3:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_2d(arr[0], ax, args, fig)
        ax.set_title(f"{title}  [0]  (use --grid or --index N for other slices)")
    else:
        return None, f"skip {path}: cannot plot {arr.ndim}-D array (use --slice)"
    return fig, info


def output_path(src: Path, out_dir: Path | None) -> Path:
    return (out_dir / f"{src.stem}.png") if out_dir else src.with_suffix(".png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, nargs="*",
                   help="input .npy file(s); if omitted, all *.npy in CWD")
    p.add_argument("--slice", default="", metavar="i,j,...",
                   help="reduce leading axes by indexing (e.g. '0,2')")
    p.add_argument("--index", type=int, default=None,
                   help="for 3D arrays, pick this leading index")
    p.add_argument("--grid", action="store_true",
                   help="for 3D arrays, tile all leading slices instead of selecting one")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--abs", action="store_true", help="plot |data|")
    p.add_argument("--log", action="store_true", help="LogNorm color scale (positive only)")
    p.add_argument("--symlog", action="store_true", help="SymLogNorm — handles negatives")
    p.add_argument("--save", type=Path, default=None,
                   help="single-input mode: save figure to this path instead of showing")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="batch mode: write <stem>.png into this dir (default: alongside source)")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    paths: list[Path] = list(args.path) if args.path else sorted(Path(".").glob("*.npy"))
    if not paths:
        sys.exit("no .npy files (pass paths or run in a directory containing some)")

    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit("missing: " + ", ".join(str(p) for p in missing))

    batch = len(paths) > 1 or args.out_dir is not None
    if batch and args.save is not None:
        sys.exit("--save is for single-file mode; use --out-dir for batch")
    if batch:
        matplotlib.use("Agg")  # headless: no display needed
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = 0
    for path in paths:
        fig, info = render(path, args)
        if fig is None:
            print(info, file=sys.stderr)
            n_skip += 1
            continue

        if batch:
            out = output_path(path, args.out_dir)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"{info} -> {out}", file=sys.stderr)
            plt.close(fig)
        elif args.save:
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"{info} -> {args.save}", file=sys.stderr)
        else:
            print(info, file=sys.stderr)
            plt.show()
        n_ok += 1

    if batch:
        print(f"done: {n_ok} converted, {n_skip} skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
