#!/usr/bin/env python3
"""Convert a DDFacet DDMSC .DicoModel pickle to .npy history files.

Walks DicoSMStacked["Comp"][iScale][(x, y)] and writes:
  historical_peak_coords.npy   (N, 2) int32   -- (row, col) per component
  historical_scales.npy        (N,)   int32   -- scale index per component

Run with /home/hugo/Projects/fast-deconv/.venv/bin/python (numpy required).
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dicomodel", type=Path, help="path to .DicoModel pickle")
    ap.add_argument("out_dir", type=Path, help="directory to write the two .npy files")
    args = ap.parse_args()

    with args.dicomodel.open("rb") as f:
        D = pickle.load(f)

    comp = D.get("Comp", {})
    coords: list[tuple[int, int]] = []
    scales: list[int] = []
    for iScale, sd in comp.items():
        if not isinstance(iScale, (int, np.integer)):
            continue
        for key in sd.keys():
            if key == "NumComps":
                continue
            x, y = key
            coords.append((int(x), int(y)))
            scales.append(int(iScale))

    coords_arr = (np.asarray(coords, dtype=np.int32)
                  if coords else np.zeros((0, 2), dtype=np.int32))
    scales_arr = np.asarray(scales, dtype=np.int32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "historical_peak_coords.npy", coords_arr)
    np.save(args.out_dir / "historical_scales.npy", scales_arr)

    print(f"wrote {len(scales_arr)} components -> {args.out_dir}/", file=sys.stderr)
    if len(scales_arr):
        unique, counts = np.unique(scales_arr, return_counts=True)
        print("  per-scale counts:", dict(zip(unique.tolist(), counts.tolist())),
              file=sys.stderr)


if __name__ == "__main__":
    main()
