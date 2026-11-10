#!/usr/bin/env python3
"""Was a past pheasy run hit by the 27-translation minimum-image bug?

[FIX WS-IMAGES] Before the fix, NeighborList.wigner_seitz searched only the
supercell translations {-1,0,1}^3.  For a supercell basis far from reduced
(e.g. a symmetric rhombohedral primitive cell repeated N x N x 1) it took a
farther image for some (central atom, supercell atom) pairs.  Read-only: this
script builds the same supercell pheasy builds (POSCAR + --dim) and reports
those pairs, their true distances and what they touched:

* cutoff selection: a pair truly inside --c2 / --c3 but measured outside was
  dropped from the cluster space (and the reverse);
* --rasr BH / H / BHH: the invariance constraints used the wrong r_ij;
* ShengBTE FORCE_CONSTANTS_3RD export: triplet vectors from the wrong image.

    python tools/ws_audit.py POSCAR --dim 4 4 1 [--c2 7.0] [--c3 3.88] [--json out.json]

Run it with the python that runs pheasy (pheasy_gpu must be importable).
"""
import argparse
import itertools
import json
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("poscar")
    ap.add_argument("--dim", nargs=3, type=int, required=True)
    ap.add_argument("--c2", type=float, default=None, help="FC2 cutoff (A) of the run")
    ap.add_argument("--c3", type=float, default=None, help="FC3 cutoff (A) of the run")
    ap.add_argument("--json", default=None, help="write the summary here")
    args = ap.parse_args(argv)

    from ase.io import read
    from pheasy_gpu.structure.atoms import Atoms, create_supercell

    prim = read(args.poscar)
    sc = create_supercell(Atoms(aseatoms=prim), np.array(args.dim))
    sc.set_pmap()
    cell = sc.cell.real
    recip = np.linalg.norm(np.linalg.inv(cell), axis=0)
    box27 = np.array(list(itertools.product((-1.0, 0.0, 1.0), repeat=3)))
    eps = 1e-4
    rows = []
    for i in sc.pmap:
        x1 = sc.scaled_positions[i]
        for j, x2 in enumerate(sc.scaled_positions):
            old = np.linalg.norm((x2 + box27 - x1) @ cell, axis=1).min()
            base = -np.round(x2 - x1)
            r0 = np.linalg.norm((x2 + base - x1) @ cell)
            reach = np.ceil(recip * (r0 + eps) + 0.5).astype(int)
            t = np.array(list(itertools.product(
                *[np.arange(-n, n + 1) + b for n, b in zip(reach, base)])))
            true = np.linalg.norm((x2 + t - x1) @ cell, axis=1).min()
            if old > true + eps:
                rows.append((int(i), int(j), float(true), float(old)))

    L = np.linalg.norm(cell, axis=1)
    out = {"poscar": args.poscar, "dim": args.dim,
           "supercell_lengths": [round(float(x), 3) for x in L],
           "pairs_checked": int(len(sc.pmap) * len(sc)),
           "pairs_wrong_image": len(rows),
           "min_true_distance_of_wrong": min((r[2] for r in rows), default=None)}
    for name, cut in (("c2", args.c2), ("c3", args.c3)):
        if cut is not None:
            out["dropped_inside_" + name] = sum(1 for r in rows if r[2] < cut <= r[3])
            out["wrong_geometry_inside_" + name] = sum(1 for r in rows if r[3] < cut)
    out["verdict"] = ("UNAFFECTED: the old search found every nearest image"
                      if not rows else
                      "AFFECTED: --rasr BH/H/BHH constraints and ShengBTE fc3 exports of "
                      "this cell used wrong images; cutoff selection too where the "
                      "counts above are non-zero")
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dict(out, pairs=rows[:2000]), fh, indent=1)
    return 0 if not rows else 1


if __name__ == "__main__":
    sys.exit(main())
