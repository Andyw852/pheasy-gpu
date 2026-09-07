#!/usr/bin/env python3
"""Validate Mg2C60 fractional-coordinate data and align it to SPOSCAR."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--max-reference-error", type=float, default=1e-5)
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args()
    d = np.load(args.source / "dataset_disps.npy")
    f = np.load(args.source / "dataset_forces.npy")
    lines = (args.source / "SPOSCAR").read_text().splitlines()
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * float(lines[1])
    counts = [int(x) for x in lines[6].split()]
    natoms = sum(counts)
    target = np.array([[float(x) for x in row.split()[:3]] for row in lines[8:8 + natoms]])
    if d.shape != f.shape or d.ndim != 3 or d.shape[1:] != (natoms, 3):
        raise ValueError(f"Unexpected data shapes: {d.shape}, {f.shape}")
    if not np.isfinite(d).all() or not np.isfinite(f).all():
        raise ValueError("Input data contains nonfinite values")
    delta = target[:, None, :] - d[-1][None, :, :]
    delta -= np.round(delta)
    distances = np.linalg.norm(delta @ cell, axis=2)
    rows, permutation = linear_sum_assignment(distances)
    matched = distances[rows, permutation]
    u = d - d[-1]
    u -= np.round(u)
    u = np.ascontiguousarray(u[:-1, permutation] @ cell)
    forces = np.ascontiguousarray((f[:-1] - f[-1])[..., :][:, permutation])
    report = {
        "source": str(args.source.resolve()),
        "input_shape": list(d.shape),
        "configuration_count": len(u),
        "reference_frame": len(d) - 1,
        "reference_force_max_eV_A": float(np.abs(f[-1]).max()),
        "minimum_force_norm_frame": int(np.argmin(np.linalg.norm(f.reshape(len(f), -1), axis=1))),
        "reference_match_max_A": float(matched.max()),
        "reference_match_rms_A": float(np.sqrt(np.mean(matched ** 2))),
        "displacement_rms_A": float(np.sqrt(np.mean(u ** 2))),
        "displacement_abs_max_A": float(np.abs(u).max()),
        "target_to_source_permutation": permutation.tolist(),
        "target_species_counts": counts,
        "first_species_source_indices": permutation[:counts[0]].tolist(),
    }
    print(json.dumps(report, indent=2))
    if args.inspect_only:
        return
    if matched.max() > args.max_reference_error:
        raise ValueError(f"Reference cannot be matched to SPOSCAR: max error {matched.max():.6g} A")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("disp_matrix.pkl", u), ("force_matrix.pkl", forces)):
        with (args.output / name).open("wb") as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (args.output / "dataset_alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "ndata_total.txt").write_text(str(len(u)) + "\n")


if __name__ == "__main__":
    main()
