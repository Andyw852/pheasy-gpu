#!/usr/bin/env python3
"""Recover an independent, crystal-symmetric spring model through the full CLI.

Run with this checkout's dependencies installed (an editable install is not
required). Each stage uses sys.executable -m pheasy_gpu.run_pheasy. --gpu
requires CUDA and verifies that the GPU least-squares implementation actually
ran; it must not silently pass by falling back to the CPU.

Examples:
    python dev/test_cli_harmonic_chain.py
    python dev/test_cli_harmonic_chain.py --gpu --workdir /tmp/si_chain_gpu
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np
from ase.build import bulk
from ase import Atoms as ASEAtoms
from ase.io import write


ROOT = Path(__file__).resolve().parents[1]


def checkout_bootstrap(workdir):
    """Expose the flat checkout as pheasy_gpu without touching the environment."""
    bootstrap = workdir / "bootstrap"
    package = bootstrap / "pheasy_gpu"
    package.mkdir(parents=True)
    # The wrapper only records a successful GPU solve. It does not alter the
    # arguments, result, solver dispatch, or production implementation.
    package.joinpath("__init__.py").write_text(
        "from pathlib import Path\n"
        f"_root = Path({str(ROOT)!r})\n"
        "__path__ = [str(_root)]\n"
        "exec(compile((_root / '__init__.py').read_text(), str(_root / '__init__.py'), 'exec'))\n"
        "import os as _os\n"
        "if _os.environ.get('PHEASY_CLI_GPU_AUDIT'):\n"
        "    from pheasy_gpu.core import gpu_backend as _gb\n"
        "    _original_lstsq = _gb.lstsq\n"
        "    def _audited_lstsq(*args, **kwargs):\n"
        "        result = _original_lstsq(*args, **kwargs)\n"
        "        import json, torch\n"
        "        event = {'function': 'gpu_backend.lstsq', 'device': str(_gb.device()),\n"
        "                 'max_memory_allocated': torch.cuda.max_memory_allocated(_gb.device())}\n"
        "        with open(_os.environ['PHEASY_CLI_GPU_AUDIT'], 'a') as stream:\n"
        "            stream.write(json.dumps(event) + '\\n')\n"
        "        return result\n"
        "    _gb.lstsq = _audited_lstsq\n"
    )
    return bootstrap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--device", type=int, default=0, help="CUDA index within visible devices")
    ap.add_argument("--workdir", type=Path, help="empty directory for inputs, logs and JSON result")
    ap.add_argument("--configs", type=int, default=12)
    ap.add_argument("--jobs", type=int, default=2, help="sensing-matrix construction workers")
    ap.add_argument("--materialize", action="store_true", help="also test materialized OLS on CPU")
    ap.add_argument("--tolerance", type=float, default=1e-10)
    args = ap.parse_args()
    if args.configs < 2 or args.jobs < 1:
        ap.error("--configs must be >= 2 and --jobs must be positive")
    workdir = (args.workdir.resolve() if args.workdir else Path(tempfile.mkdtemp(prefix="pheasy_si_chain_")))
    if workdir.exists() and any(workdir.iterdir()):
        ap.error("--workdir must be empty")
    workdir.mkdir(parents=True, exist_ok=True)
    bootstrap = checkout_bootstrap(workdir)
    sys.path.insert(0, str(bootstrap))
    from pheasy_gpu.structure.atoms import Atoms, create_supercell

    primitive = bulk("Si", "diamond", a=5.43)
    scell = create_supercell(Atoms(aseatoms=primitive), np.array([2, 2, 2]))
    write(workdir / "POSCAR", primitive, format="vasp", direct=True)
    write(workdir / "SPOSCAR", scell, format="vasp", direct=True)
    geometry = ASEAtoms(numbers=scell.numbers, positions=scell.positions,
                        cell=scell.cell, pbc=True)
    distances = geometry.get_all_distances(mic=True)
    nearest = float(distances[distances > 1e-8].min())
    cutoff = 1.05 * nearest
    neighbors = (distances > 1e-8) & (distances < cutoff)
    assert np.all(neighbors.sum(axis=1) == 4), "diamond nearest-neighbor graph must have degree 4"
    natoms = len(scell)
    phi = -neighbors[:, :, None, None].astype(float) * np.eye(3)[None, None]
    phi[np.arange(natoms), np.arange(natoms)] = neighbors.sum(axis=1)[:, None, None] * np.eye(3)
    np.testing.assert_array_equal(phi.sum(axis=1), 0)
    np.testing.assert_array_equal(phi, phi.transpose(1, 0, 3, 2))
    rng = np.random.default_rng(20260907)
    u = rng.normal(0, 0.01, (args.configs, natoms, 3))
    # The truth is generated from geometry alone; no sensing/null-space/IFC
    # expansion code participates in the reference force calculation.
    forces = -np.einsum("ijab,njb->nia", phi, u, optimize=True)
    for filename, data in (("disp_matrix.pkl", u), ("force_matrix.pkl", forces)):
        with (workdir / filename).open("wb") as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(workdir / "fc2_expected.npy", phi)
    env = {key: value for key, value in os.environ.items() if not key.startswith("PHEASY_")}
    env.update({"PYTHONPATH": str(bootstrap), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "PHEASY_N_JOBS": str(args.jobs), "PHEASY_DOT_THREADS": "1",
                "PHEASY_SM_DTYPE": "float64", "PHEASY_USE_GPU": "1" if args.gpu else "0",
                "PHEASY_OLS_TWOLEVEL": "0" if args.gpu or args.materialize else "1",
                "PHEASY_OLS_ATOL": "1e-13", "PHEASY_OLS_BTOL": "1e-13",
                "PHEASY_LSQR_ATOL": "1e-13", "PHEASY_LSQR_BTOL": "1e-13"})
    if args.gpu:
        env["PHEASY_GPU_DEVICE"] = str(args.device)
        env["PHEASY_GPU_DEVICES"] = str(args.device)
        env["PHEASY_CLI_GPU_AUDIT"] = str(workdir / "gpu_calls.jsonl")
        probe = subprocess.run([sys.executable, "-c", "from pheasy_gpu.core import gpu_backend as g; "
                                "assert g.available(), 'CUDA unavailable'; print(g.device())"],
                               cwd=workdir, env=env, capture_output=True, text=True)
        if probe.returncode:
            raise RuntimeError(probe.stdout + probe.stderr)
    base = [sys.executable, "-m", "pheasy_gpu.run_pheasy", "--dim", "2", "2", "2", "-w", "2",
            "--c2", str(cutoff), "--eps", "0.001"]
    stages = [("cluster", ["-s"]), ("constraints", ["-c"]),
              ("sensing", ["-d", "--ndata", str(args.configs), "--disp_file"]),
              ("fit", ["-f", "--ndata", str(args.configs), "--disp_file", "--full_ifc", "-l", "OLS", "--hdf5"])]
    timings = {}
    for name, flags in stages:
        print(f"{name}: {workdir / (name + '.log')}", flush=True)
        start = time.perf_counter()
        with (workdir / (name + ".log")).open("w") as stream:
            result = subprocess.run(base + flags, cwd=workdir, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        timings[name] = time.perf_counter() - start
        if result.returncode:
            tail = (workdir / (name + ".log")).read_text().splitlines()[-35:]
            raise RuntimeError(f"CLI {name} failed with status {result.returncode}\n" + "\n".join(tail))
    with h5py.File(workdir / "fc2.hdf5", "r") as stream:
        key = next(key for key in ("force_constants", "fc2") if key in stream)
        fitted = stream[key][:]
    assert fitted.shape == phi.shape, (fitted.shape, phi.shape)
    prediction = -np.einsum("ijab,njb->nia", fitted, u, optimize=True)
    max_error = float(np.max(np.abs(fitted - phi)))
    relative_error = max_error / float(np.max(np.abs(phi)))
    force_relative_error = float(np.linalg.norm(prediction - forces) / np.linalg.norm(forces))
    asr_error = float(np.abs(fitted.sum(axis=1)).max())
    gpu_events = []
    if args.gpu and (workdir / "gpu_calls.jsonl").exists():
        gpu_events = [json.loads(line) for line in (workdir / "gpu_calls.jsonl").read_text().splitlines()]
    report = {"checkout": str(ROOT), "python": sys.executable, "workdir": str(workdir),
              "gpu_requested": args.gpu, "gpu_calls": gpu_events, "natoms": natoms,
              "configs": args.configs, "spring_eV_A2": 1.0, "nearest_A": nearest, "cutoff_A": cutoff,
              "fc2_max_absolute_error": max_error, "fc2_max_relative_error": relative_error,
              "force_relative_error": force_relative_error, "asr_max_absolute_error": asr_error,
              "stage_seconds": timings}
    passed = (relative_error < args.tolerance and force_relative_error < args.tolerance
              and asr_error < args.tolerance and (not args.gpu or bool(gpu_events)))
    report["passed"] = passed
    (workdir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert passed, f"harmonic CLI recovery failed; inspect {workdir / 'result.json'}"


if __name__ == "__main__":
    main()
