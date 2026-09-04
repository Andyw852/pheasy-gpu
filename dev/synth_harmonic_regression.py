#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic harmonic recovery regression (pheasy-gpu).

Certifies the WHOLE harmonic chain -- cluster space -> null space -> sensing
matrix -> force read-in -> OLS -> ASR-as-constraint -> fc2 write -- with an
INDEPENDENT real-space force generator (never F = SM @ coef, which is in the
column space by construction and cannot catch SM/indexing bugs).

Method:
  1. Build a known Phi2 inside the cutoff from a reference fc2 (or analytic
     springs) + ASR on the truncated basis.
  2. Generate N random displacements u and forces F[n,i] = -sum_j Phi[i,j] u[n,j]
     directly in real space.
  3. Fit with pheasy (order 2 only) and assert the recovered fc2 equals Phi
     to machine precision (~1e-10).

Measured (Si 4x4x4, 40 configs, cutoff 5.45 A): rel recovery max = 1.2e-15.
Run from a dir containing POSCAR (unit cell) + a pheasy-gpu install:
    python dev/synth_harmonic_regression.py POSCAR [--dim 4 4 4 --c2 5.45 --n 40]
"""
import argparse
import h5py
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def build_supercell(poscar, dim):
    from phonopy import Phonopy
    from phonopy.interface.vasp import read_vasp
    uc = read_vasp(poscar)
    ph = Phonopy(uc, supercell_matrix=np.diag(dim))
    sc = ph.supercell
    pos = np.array(sc.positions)
    cell = np.array(sc.cell)
    n = len(sc)
    dmat = np.zeros((n, n))
    for i in range(n):
        best = np.full(n, 1e9)
        for tx in (-1, 0, 1):
            for ty in (-1, 0, 1):
                for tz in (-1, 0, 1):
                    T = tx * cell[0] + ty * cell[1] + tz * cell[2]
                    v = pos - pos[i][None] + T
                    best = np.minimum(best, np.linalg.norm(v, axis=1))
        dmat[i] = best
    np.fill_diagonal(dmat, 0)
    return ph, dmat


def known_phi(symfc_fc2_hdf5, dmat, cutoff):
    """Reference harmonic model: real fc2 restricted inside cutoff + ASR."""
    with h5py.File(symfc_fc2_hdf5) as f:
        key = 'force_constants' if 'force_constants' in f else list(f.keys())[0]
        sym = f[key][:]
    Phi = sym.copy()
    Phi[dmat >= cutoff] = 0.0
    n = Phi.shape[0]
    for i in range(n):
        Phi[i, i] -= Phi[i].sum(axis=0)   # ASR on the truncated basis
    return Phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('poscar')
    ap.add_argument('--dim', nargs=3, type=int, default=[4, 4, 4])
    ap.add_argument('--c2', type=float, default=5.45)
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--ref-fc2', default=None,
                    help='optional real fc2 hdf5 to seed Phi_known (else analytic springs)')
    ap.add_argument('--pheasy', default='pheasy-gpu')
    args = ap.parse_args()
    ph, dmat = build_supercell(args.poscar, args.dim)
    n = len(ph.supercell)
    if args.ref_fc2:
        Phi = known_phi(args.ref_fc2, dmat, args.c2)
    else:
        rng = np.random.default_rng(3)
        Phi = np.zeros((n, n, 3, 3))
        iis, jjs = np.where((dmat < args.c2) & (dmat > 1e-6))
        for i, j in zip(iis, jjs):
            M = rng.normal(0, 1, (3, 3))
            M = (M + M.T) / 2
            Phi[i, j] = M
        for i in range(n):
            Phi[i, i] = -Phi[i].sum(axis=0)
    # independent real-space force generation
    rng = np.random.default_rng(7)
    u = rng.normal(0, 0.01, (args.n, n, 3))
    F = -np.einsum('ijab,njb->nia', Phi, u, optimize=True)
    tmp = tempfile.mkdtemp(prefix='synth_harm_')
    shutil.copy(args.poscar, os.path.join(tmp, 'POSCAR'))
    for fn, arr in (('disp_matrix.pkl', u), ('force_matrix.pkl', F)):
        pickle.dump(arr, open(os.path.join(tmp, fn), 'wb'), protocol=4)
    base = '%s --dim %d %d %d -w 2 --c2 %s --eps 0.001' % (
        args.pheasy, args.dim[0], args.dim[1], args.dim[2], args.c2)
    cmds = [
        base + ' -s',
        base + ' -c',
        base + ' -d --ndata %d --disp_file' % args.n,
        base + ' -f --ndata %d --disp_file --full_ifc -l OLS --hdf5' % args.n,
    ]
    for cmd in cmds:
        r = subprocess.run(cmd.split(), cwd=tmp, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            sys.exit('pheasy step failed: %s' % cmd)
    with h5py.File(os.path.join(tmp, 'fc2.hdf5')) as f:
        fc = f[list(f.keys())[0]][:]
    d = np.abs(fc - Phi)
    rel = d.max() / np.abs(Phi).max()
    print('recovered fc2 vs Phi_known: max|d| = %.3e   rel = %.3e' % (d.max(), rel))
    assert rel < 1e-10, 'harmonic recovery FAILED (rel %.3e)' % rel
    print('PASS: harmonic chain certified to machine precision')


if __name__ == '__main__':
    main()
