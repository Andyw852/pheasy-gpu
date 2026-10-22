"""Classes and functions related to symmetry operations."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Changpeng Lin
# All rights reserved.

__all__ = [
    "Symmetrizer",
    "get_primitive_cell",
    "get_spacegroup",
    "get_symmetry",
    "get_ir_reciprocal_grid",
    "spacegroup_record",
    "symmetry_summary",
    "symmetry_warning",
]


import os
import re

import numpy as np

try:
    import spglib as spg
except ImportError:
    try:
        import phonopy.structure.spglib as spg
    except ImportError:
        from pyspglib import spglib as spg


class Symmetrizer(object):
    """Symmetrizer for seveal physical quantities.

    Space group is used to symmetrize Born-effective charges
    with acoustic sum rules included.
    
    """

    pass


def get_primitive_cell(struct, symprec=1e-5, angle_tolerance=-1.0):
    """Find lattice vectors of primitive unit cell for a given structure.

    This is a wrapper for spglib.find_primitive to find the
    primitive unit cell.

    Parameters
    ----------
    struct : pheasy.Atoms
        Input structure.
    symprec : float
        Tolerance for finding primitive unit cell.
    angle_tolerance : float
        Tolerance for determing angle of cell.

    Returns
    -------
    A tuple of (lattice, positions, numbers).

    """
    struct_tuple = struct.to_spglib_tuple()
    unit_cell_tuple = spg.find_primitive(
        struct_tuple, symprec=symprec, angle_tolerance=angle_tolerance
    )
    if unit_cell_tuple is None:
        return None
    else:
        return unit_cell_tuple


def get_spacegroup(struct, symprec=1e-5, angle_tolerance=-1.0, symbol_type=0):
    """Return space group symbol and number as a string for a given structure.

    This is a wrapper for spglib.get_spacegroup.

    Parameters
    ----------
    struct : pheasy.Atoms
        Input structure.
    symprec : float
        Tolerance for determining symmetry.
    angle_tolerance : float
        Tolerance for determing angle of cell.
    symbol_type : int
        Space group symbol type.
        0 for international symbol.
        1 for schoenflies symbol.

    Returns
    -------
    A tuple of (symbol, number).

    """
    struct_tuple = struct.to_spglib_tuple()
    return spg.get_spacegroup(
        struct_tuple,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        symbol_type=symbol_type,
    )


def get_symmetry(
    struct, symprec=1e-5, angle_tolerance=-1.0, is_magnetic=False, mag_symprec=-1.0
):
    """Find and return symmetry operations.

    This is a wrapper for spglib.get_symmetry. The docstring here has been
    adapted from the spglib one, see:
    https://github.com/spglib/spglib/blob/develop/python/spglib/spglib.py

    Parameters
    ----------
    struct : pheasy.Atoms
        Input structure
    symprec : float
        Tolerance for determining symmetry.
    angle_tolerance : float
        Tolerance for determing angle of cell.
    is_magnetic : bool
        True to consider magnetic symmetry with the provided magmom data.
        In case of collinear calculation, an array with the shape (natoms,)
        should be provided. In case of non-collinear calculation, an array
        with the shape (natoms,3) should be provided.
    mag_symprec : float
        Tolerance for magnetic symmetry search in the unit of magmoms.
        If not specified, use the same value as symprec.
    

    Returns
    -------
    dict
        Rotation parts and translation parts of symmetry operations
        represented with respect to basis vectors and atom index
        mapping by symmetry operations.
        'rotations' : numpy.ndarray
            Rotation (matrix) parts of symmetry operations
            shape=(num_operations, 3, 3), order='C', dtype='intc'
        'translations' : numpy.ndarray
            Translation (vector) parts of symmetry operations
            shape=(num_operations, 3), dtype='double'
        'equivalent_atoms' : numpy.ndarray
            shape=(num_atoms, ), dtype='intc'
    
    """
    cell, scaled_positions, types = struct.to_spglib_tuple()
    if is_magnetic:
        # TODO
        magmoms = struct.get_initial_magnetic_moments()
        struct_tuple = (cell, scaled_positions, types, magmoms)
        return spg.get_magnetic_symmetry(
            struct_tuple,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            mag_symprec=mag_symprec,
            with_time_reversal=True,
        )
    else:
        struct_tuple = (cell, scaled_positions, types)
        return spg.get_symmetry(
            struct_tuple, symprec=symprec, angle_tolerance=angle_tolerance,
        )


def get_ir_reciprocal_grid(struct, grid_size, is_shift=[0, 0, 0]):
    """Generate irreducible k/q-points of the given uniform grid.

    This is a wrapper for spglib.get_ir_reciproca_mesh. The docstring 
    here has been adapted from the spglib one, see:
    https://github.com/spglib/spglib/blob/develop/python/spglib/spglib.py

    Parameters
    ----------
    struct : pheasy.Atoms
        Input structure
    grid_size : list or numpy.array
        Input uniform grid where irreducible k/q-points is searched.
        All of elements should be integer.
    is_shift : list or numpy.array, optional
        When is_shift is set for each reciprocal primitive axis, 
        the mesh is shifted along the axis in half of adjacent mesh 
        points irrespective of the mesh numbers. Defaults to [0,0,0].


    Returns:
        ir_grid : numpy.array
            An array of irreducible k/q-points in scaled positions,
            ranging from [-0.5, 0.5) and the shape is (nipts, 3).
        weights : numpy.array
            The weight of each irreducible points.

    """
    struct_tuple = struct.to_spglib_tuple()

    mapping, ir_grid = spg.get_ir_reciprocal_mesh(
        grid_size, struct_tuple, is_shift=is_shift
    )
    ir_inds = np.unique(mapping)

    weights = np.zeros_like(mapping)
    for pt in mapping:
        weights[pt] += 1
    ir_grid = np.array(ir_grid) / grid_size
    ir_grid = np.where(ir_grid >= 0.5, ir_grid - 1.0, ir_grid)

    return ir_grid[ir_inds], weights[ir_inds]


def spacegroup_record(spacegroup, n_operations, symprec):
    """Space group found at one tolerance, as a JSON-ready dict.

    ``spacegroup`` is spglib's "Symbol (number)" string (or None when spglib
    found none); ``n_operations`` the number of symmetry operations.
    """
    symbol, number = None, None
    if spacegroup:
        m = re.match(r"^\s*(.*?)\s*\((\d+)\)\s*$", str(spacegroup))
        if m:
            symbol, number = m.group(1), int(m.group(2))
        else:
            symbol = str(spacegroup).strip()
    return {
        "symprec": float(symprec),
        "spacegroup": None if spacegroup is None else str(spacegroup),
        "symbol": symbol,
        "number": number,
        "n_operations": None if n_operations is None else int(n_operations),
    }


def _find_spacegroup_record(struct, symprec, is_magnetic=False):
    spacegroup = get_spacegroup(struct, symprec=symprec)
    ops = get_symmetry(struct, symprec=symprec, is_magnetic=is_magnetic)
    n_ops = None if ops is None else int(np.asarray(ops["translations"]).shape[0])
    return spacegroup_record(spacegroup, n_ops, symprec)


def symmetry_summary(struct, symprec, spacegroup, n_operations,
                     is_magnetic=False, probe_symprec=None):
    """Symmetry used by the fit plus a probe at a looser tolerance.

    A structure with a small numerical distortion (relaxation noise of
    1e-5..1e-4 A) is found in a SUBGROUP at a tight --symprec, and every force
    constant is then constrained by that subgroup only.  A later step run at a
    looser tolerance (phono3py/kappa at 1e-3) symmetrizes the result and hides
    the problem.  The probe re-detects the space group at ``probe_symprec``
    (default PHEASY_SYMPREC_PROBE, 1e-3; <= 0 disables it) when that is looser
    than ``symprec``.

    Returns a dict: the fit record (spacegroup_record) under "fit", the probe
    record (or None) under "probe", and "probe_more_symmetric" (True when the
    probe finds more operations).
    """
    if probe_symprec is None:
        probe_symprec = float(os.environ.get("PHEASY_SYMPREC_PROBE", "1e-3"))
    probe_symprec = float(probe_symprec)
    out = {
        "symprec": float(symprec),
        "fit": spacegroup_record(spacegroup, n_operations, symprec),
        "probe": None,
        "probe_more_symmetric": False,
    }
    if probe_symprec > 0 and probe_symprec > float(symprec):
        probe = _find_spacegroup_record(struct, probe_symprec, is_magnetic)
        out["probe"] = probe
        out["probe_more_symmetric"] = bool(
            probe["n_operations"] is not None and n_operations is not None
            and probe["n_operations"] > int(n_operations))
    return out


def symmetry_warning(summary):
    """Warning text when the probe tolerance finds more symmetry, else None."""
    if not summary or not summary.get("probe_more_symmetric"):
        return None
    fit, probe = summary["fit"], summary["probe"]
    return (
        "!!! Symmetry depends on the tolerance: --symprec %g finds %s with %d "
        "operations, but symprec %g finds %s with %d.  The force constants are "
        "constrained by the LOWER symmetry only (more free parameters, IFCs that "
        "break the true symmetry); a later step at the looser tolerance would "
        "symmetrize them and hide it.  Use --symprec %g here and the same value "
        "downstream, or symmetrize the structure before generating the "
        "displacements.  Changing --symprec invalidates the cluster space: rerun "
        "from the cluster stage (-s). !!!"
        % (fit["symprec"], fit["spacegroup"], fit["n_operations"],
           probe["symprec"], probe["spacegroup"], probe["n_operations"],
           probe["symprec"]))
