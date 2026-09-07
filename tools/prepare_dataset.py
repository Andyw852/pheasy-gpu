#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a displacement/force dataset into the pickles pheasy expects.

pheasy's ``--disp_file`` path reads ``disp_matrix.pkl`` (Cartesian
displacements, shape ``(ndata, natoms, 3)``, Angstrom) and ``force_matrix.pkl``
(forces, same shape, eV/A). This script builds both from a pair of .npy files.

The displacement array may hold either

  * Cartesian displacements already, or
  * **fractional coordinates** of each configuration -- in which case pass
    ``--frac``; the reference frame (``--ref``, default the last one) is
    subtracted, minimum-image wrapped and converted with the supercell lattice.
    Add ``--align-reference`` to match reference atoms to SPOSCAR positions
    and apply the same permutation to displacements and forces.

The corresponding residual forces of the reference frame are subtracted from
every configuration unless ``--no-subtract-residual`` is given.

.. warning::
   Without --align-reference, input arrays MUST be in SPOSCAR atom order.
   A pheasy-generated
   SPOSCAR uses create_supercell order (per-primitive-atom blocks: all
   images of primitive atom 0, then atom 1, ...); ASE Atoms.repeat()
   interleaves per image instead, and data prepared that way silently
   scrambles every atom but the first (the origin image). The fit then
   looks sane for the on-site IFC yet carries a ~50% residual --
   run_pheasy now prints a post-fit corr check that flags it (AGENTS.md).

Usage
-----
    python3 tools/prepare_dataset.py SPOSCAR dataset_disps.npy dataset_forces.npy --frac
    python3 tools/prepare_dataset.py SPOSCAR dataset_disps.npy dataset_forces.npy --frac --align-reference
"""
import argparse
import json
import pickle

import numpy as np


def read_lattice(sposcar):
    # ASE handles VASP4/5, negative-volume scaling, and Cartesian coordinates.
    from ase.io import read
    return np.asarray(read(sposcar, format="vasp").cell)


def align_fractional_reference(reference, sposcar, tolerance=1e-5):
    """Return target-to-source indices for an exact SPOSCAR atom mapping.

    Matching uses periodic Cartesian distances and fails if no bijection is
    within tolerance. The arrays themselves carry no chemical identities;
    SPOSCAR must describe the same structure as the supplied reference.
    """
    from ase.geometry import find_mic
    from ase.io import read
    from scipy.optimize import linear_sum_assignment

    atoms = read(sposcar, format="vasp")
    target = atoms.get_scaled_positions()
    if reference.shape != target.shape:
        raise ValueError("Reference atom count does not match SPOSCAR")
    delta = target[:, None, :] - reference[None, :, :]
    delta -= np.round(delta)
    _, distances = find_mic((delta @ atoms.cell).reshape(-1, 3),
                            atoms.cell, pbc=True)
    distances = distances.reshape(len(target), len(target))
    rows, permutation = linear_sum_assignment(distances)
    error = float(distances[rows, permutation].max())
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Alignment tolerance must be finite and positive")
    if error > tolerance:
        raise ValueError("Reference does not match SPOSCAR within %.6g A "
                         "(maximum matching error %.6g A)" % (tolerance, error))
    # Duplicate reference sites would make a permutation chemically ambiguous.
    if np.any(np.sum(distances <= tolerance, axis=1) != 1):
        raise ValueError("Reference mapping is ambiguous within alignment tolerance")
    return permutation, error


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sposcar")
    ap.add_argument("disps_npy")
    ap.add_argument("forces_npy")
    ap.add_argument("--frac", action="store_true",
                    help="input holds fractional coordinates, not displacements")
    ap.add_argument("--ref", type=int, default=-1,
                    help="index of the undisplaced reference frame (default: last)")
    ap.add_argument("--align-reference", action="store_true",
                    help="with --frac, reorder both arrays to match SPOSCAR")
    ap.add_argument("--alignment-tolerance", type=float, default=1e-5,
                    help="maximum reference matching error in Angstrom")
    ap.add_argument("--alignment-out", default="dataset_alignment.json",
                    help="permutation report written with --align-reference")
    ap.add_argument("--no-subtract-residual", action="store_true")
    ap.add_argument("--disp-out", default="disp_matrix.pkl")
    ap.add_argument("--force-out", default="force_matrix.pkl")
    args = ap.parse_args()
    if args.align_reference and not args.frac:
        ap.error("--align-reference requires --frac")

    d = np.load(args.disps_npy)
    f = np.load(args.forces_npy)
    if d.shape != f.shape or d.ndim != 3 or d.shape[2] != 3:
        raise SystemExit("expected matching (ndata, natoms, 3) arrays, got "
                         "%s and %s" % (d.shape, f.shape))
    if not len(d) or not -len(d) <= args.ref < len(d):
        raise SystemExit("reference frame index is outside the dataset")
    if not np.isfinite(d).all() or not np.isfinite(f).all():
        raise SystemExit("input arrays contain nonfinite values")
    # the atom count must match the SPOSCAR (order too -- see docstring)
    try:
        _n_spos = sum(int(x) for x in open(args.sposcar).read().splitlines()[6].split())
        if _n_spos != d.shape[1]:
            raise SystemExit("SPOSCAR has %d atoms but the dataset has %d; "
                             "atom order/count must match the SPOSCAR" % (_n_spos, d.shape[1]))
    except (IndexError, ValueError):
        print("warning: could not parse the SPOSCAR atom count; skipping check")
    alignment = None
    if args.align_reference:
        permutation, error = align_fractional_reference(
            d[args.ref], args.sposcar, args.alignment_tolerance)
        d, f = d[:, permutation], f[:, permutation]
        alignment = {
            "reference_frame": args.ref % len(d),
            "reference_match_max_A": error,
            "target_to_source_permutation": permutation.tolist(),
        }
        print("reference aligned to SPOSCAR: maximum error %.6g A" % error)

    if args.frac:
        cell = read_lattice(args.sposcar)
        ref = d[args.ref]
        u = d - ref[None]
        u = u - np.round(u)                      # minimum image
        keep = [i for i in range(d.shape[0]) if i != args.ref % d.shape[0]]
        U = u[keep] @ cell
        F = f[keep]
        if not args.no_subtract_residual:
            F = F - f[args.ref][None]
    else:
        U, F = d, f
        if not args.no_subtract_residual:
            U = np.delete(U, args.ref % d.shape[0], axis=0)
            F = np.delete(F, args.ref % d.shape[0], axis=0) - f[args.ref][None]

    print("displacements %s  rms=%.4f A  max=%.4f A" %
          (U.shape, np.sqrt((U ** 2).mean()), np.abs(U).max()))
    print("forces        %s  max=%.4f eV/A" % (F.shape, np.abs(F).max()))
    if np.abs(U).max() > 0.5:
        print("WARNING: displacements above 0.5 A -- did you mean --frac?")

    with open(args.disp_out, "wb") as fh:
        pickle.dump(np.ascontiguousarray(U, dtype=np.float64), fh)
    with open(args.force_out, "wb") as fh:
        pickle.dump(np.ascontiguousarray(F, dtype=np.float64), fh)
    with open("ndata_total.txt", "w") as fh:
        fh.write(str(U.shape[0]))
    if alignment is not None:
        with open(args.alignment_out, "w") as fh:
            json.dump(alignment, fh, indent=2)
            fh.write("\n")
    print("wrote %s, %s, ndata_total.txt (--ndata %d)"
          % (args.disp_out, args.force_out, U.shape[0]))


if __name__ == "__main__":
    main()
