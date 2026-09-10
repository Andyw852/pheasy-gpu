#!/usr/bin/env python3
"""Read-only cache regression/benchmark for the opt-in resident CUDA LASSO.

Examples (run this checkout inside a GPU allocation; installation optional):
  CUDA_VISIBLE_DEVICES=2 python dev/validate_resident_lasso.py DATA --output tmp/run-new
  python dev/validate_resident_lasso.py --synthetic --cpu-reference --output tmp/small-new
  python dev/validate_resident_lasso.py DATA --cv-tol 1e-3 --cv-max-iter 800 --output tmp/matched-new

--device is a *logical* index within CUDA_VISIBLE_DEVICES; the mask is never
changed. CV folds can use multiple GPUs. No cache writes, dense sensing-matrix materialization,
remote actions, or overwrite of existing output directories. Full CPU refitting
is optional and can be expensive; independent CPU factorized predictions are
always checked. A CPU-only smoke run cannot certify resident CUDA dispatch.
"""
import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import warnings


def write_json(path, value):
    def convert(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        return str(item)
    path.write_text(json.dumps(value, default=convert, indent=2) + "\n")


@contextmanager
def environment(values):
    previous = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, value in previous.items():
            if value is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", nargs="?", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--cpu-only", action="store_true", help="smoke test only; never reports GPU PASS")
    p.add_argument("--debias", type=int, choices=(0, 1), default=1, help="baseline 1; 0 isolates regularized solver")
    p.add_argument("--cpu-reference", action="store_true", help="also run the entire CPU Optimizer fit")
    p.add_argument("--device", type=int, default=0, help="primary logical CUDA device, default 0")
    p.add_argument("--devices", default=None, help="comma-separated logical CUDA devices for dynamic CV folds")
    p.add_argument("--ngpu", type=int, default=None, help="maximum number of resident CV GPUs")
    p.add_argument("--group-size", "--rows-per-config", dest="group_size", type=int, default=None,
                   help="optional CV group size; default None preserves historical random row splits")
    p.add_argument("--cv-tol", type=float, default=1e-6, help="strict default; historical CV uses 1e-3")
    p.add_argument("--cv-max-iter", type=int, default=20000, help="strict default; historical CV uses 800")
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--force-key", help="array in fm1d.npz; default F or unique nonmetadata key")
    p.add_argument("--prediction-rtol", type=float, default=1e-9)
    p.add_argument("--fit-rtol", type=float, default=1e-3, help="relative CPU/GPU refit prediction tolerance")
    return p


def load_data(args, np, sp):
    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        prime = sp.csr_matrix(rng.normal(size=(180, 30)))
        ns = sp.csr_matrix(rng.normal(size=(30, 16)))
        truth = np.zeros(16)
        truth[[1, 4, 9]] = [1.2, -0.7, 0.3]
        forces = prime @ (ns @ truth) + rng.normal(scale=1e-3, size=180)
        return prime, ns, forces, {"synthetic": True, "seed": args.seed}
    data = args.data.resolve(strict=True)
    paths = [data / name for name in ("sm_prime.npz", "ns_harm.npz", "fm1d.npz")]
    provenance = {str(p): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths}
    prime = sp.load_npz(paths[0])
    ns = sp.load_npz(paths[1])
    for path, matrix in ((paths[0], prime), (paths[1], ns)):
        provenance[str(path)].update(dtype=str(matrix.dtype), shape=matrix.shape, nnz=int(matrix.nnz))
    prime = prime.astype(np.float64).tocsr()
    ns = ns.astype(np.float64).tocsr()
    # Harmonic-only caches are valid. Include anharmonic constraints only when
    # required by the stored matrix width, never guess/truncate rows or columns.
    if prime.shape[1] != ns.shape[0]:
        extra = data / "ns_anharm3.npz"
        provenance[str(extra)] = {"bytes": extra.stat().st_size, "mtime_ns": extra.stat().st_mtime_ns}
        ns = sp.block_diag([ns, sp.load_npz(extra)], format="csr", dtype=np.float64)
    with np.load(paths[2], allow_pickle=False) as archive:
        keys = [key for key in archive.files if not key.startswith("_")]
        key = args.force_key or ("F" if "F" in keys else keys[0] if len(keys) == 1 else None)
        if key is None:
            raise ValueError("ambiguous fm1d.npz arrays; specify --force-key: " + repr(keys))
        raw_forces = archive[key]
        provenance[str(paths[2])].update(key=key, dtype=str(raw_forces.dtype), shape=raw_forces.shape)
        forces = np.asarray(raw_forces, dtype=np.float64).ravel()
    provenance["precision"] = {"computation_dtype": "float64",
        "note": "Stored factors/forces are promoted to FP64; this is not identical arithmetic to a legacy FP32 run. CPU reference uses the same FP64 inputs, not legacy FP32."}
    return prime, ns, forces, provenance


def relative_error(a, b, np):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def diagnostics(optimizer):
    # Keep public diagnostics, including newer backend fields, without copying
    # private cached matrices/tensors into JSON or relying on iteration counts.
    names = ("backend_", "dispatch_info_", "resident_solver_info_", "gpu_resident_info_",
             "cv_solver_info_", "regularized_solver_info_", "debias_solver_info_", "solver_diagnostics_",
             "alpha_grid_info_", "alpha_auto_",
             "alpha_", "alphas_", "mse_path_", "n_iter_", "cv_devices_", "cv_fold_devices_")
    model = {name: getattr(optimizer.model, name) for name in names if hasattr(optimizer.model, name)}
    results = {k: v for k, v in optimizer.results.items() if "coef" not in k}
    return {"model": model, "results": results, "metrics": optimizer.metrics}


def convergence_flags(value, path=""):
    flags = []
    if isinstance(value, dict):
        for key, item in value.items():
            location = path + "/" + str(key)
            if key in ("converged", "all_converged", "cv_converged") and isinstance(item, (bool, int)):
                flags.append({"path": location, "converged": bool(item)})
            else:
                flags.extend(convergence_flags(item, location))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            flags.extend(convergence_flags(item, path + "/" + str(i)))
    return flags


def run_fit(args, output, prime, ns, forces, np, torch, om, gpu):
    label = "gpu" if gpu else "cpu"
    env = {"PHEASY_GPU_LASSO_RESIDENT": "1" if gpu else "0",
           "PHEASY_USE_GPU": "1" if gpu else "0", "PHEASY_GPU_SM": "0",
           "PHEASY_GPU_DEVICE": str(args.device),
           "PHEASY_GPU_DEVICES": str(args.devices) if args.devices else "",
           "PHEASY_GPU_NGPU": str(args.ngpu) if args.ngpu is not None else "",
           "PHEASY_GPU_SM_DEVICES": str(args.device), "PHEASY_GPU_SM_NGPU": "1",
           "PHEASY_TWOLEVEL_CACHE_T": "0", "PHEASY_LASSO_N_JOBS": "1", "PHEASY_N_JOBS": "1",
           "PHEASY_CV_GROUP_SIZE": str(args.group_size or 0),
           "PHEASY_CV_TOL": str(args.cv_tol), "PHEASY_CV_MAX_ITER": str(args.cv_max_iter),
           "PHEASY_LASSO_DEBIAS": str(args.debias)}
    with environment(env), (output / (label + ".log")).open("x") as log, redirect_stdout(log), redirect_stderr(log):
        if gpu:
            from core import gpu_backend
            selected_devices = gpu_backend._resident_cv_devices()[:5]
            for selected_device in selected_devices:
                with torch.cuda.device(selected_device):
                    torch.empty(0, device=selected_device)
                    torch.cuda.synchronize(selected_device)
                    torch.cuda.reset_peak_memory_stats(selected_device)
        start = time.monotonic()
        operator = om.TwoLevelSM(prime, ns, dtype=np.float64)
        grid_start = time.monotonic()
        # Both current Optimizer branches expect the CLI to derive the grid.
        # Include this CPU stage in timing; do not call it resident GPU math.
        alpha = om.derive_alpha_grid(operator, forces, nalpha=20,
                                     decades=4.0, standardize=True)
        grid_seconds = time.monotonic() - grid_start
        optimizer = om.Optimizer("LASSO", nalpha=20, cv=5, tol=1e-6, max_iter=20000,
                                 alpha=alpha, alpha_auto=True, decades=4.0,
                                 standardize=True, rand_seed=args.seed, use_gpu=gpu)
        debias_records = []
        debias_seconds = 0.0
        original_debias = optimizer._debias
        original_info = om._iterative_solver_info

        def record_debias_info(*values, **kwargs):
            info = original_info(*values, **kwargs)
            debias_records.append(dict(info, stage="debias"))
            return info

        def audited_debias(*values, **kwargs):
            nonlocal debias_seconds
            debias_start = time.monotonic()
            # Instrument only the optional post-LASSO solve; never stop CV.
            om._iterative_solver_info = record_debias_info
            try:
                return original_debias(*values, **kwargs)
            finally:
                debias_seconds += time.monotonic() - debias_start
                om._iterative_solver_info = original_info

        optimizer._debias = audited_debias
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                optimizer.fit(operator, forces)
        finally:
            optimizer._debias = original_debias
            om._iterative_solver_info = original_info
        if gpu:
            for selected_device in selected_devices:
                torch.cuda.synchronize(selected_device)
        seconds = time.monotonic() - start
        coef = np.asarray(optimizer.results["coef"])
        prediction = np.asarray(optimizer.predict(operator)).ravel()
        cpu_prediction = np.asarray(prime @ (ns @ coef)).ravel()
        diag = diagnostics(optimizer)
        diag["debias_iterative_solves"] = debias_records
        pre_coef = optimizer.results.get("pre_debias_coef")
        pre_arrays = {}
        if pre_coef is not None:
            pre_coef = np.asarray(pre_coef)
            pre_prediction = np.asarray(prime @ (ns @ pre_coef)).ravel()
            pre_arrays = {"pre_debias_coefficients": pre_coef, "pre_debias_prediction": pre_prediction}
        # Normalize NumPy scalars before recursively interpreting boolean fields.
        write_json(output / (label + "_diagnostics.json"), diag)
        diag = json.loads((output / (label + "_diagnostics.json")).read_text())
        flags = convergence_flags(diag)
        cv_flags = [flag for flag in flags if "cv" in flag["path"].lower()]
        np.savez(output / (label + "_arrays.npz"), coefficients=coef,
                 prediction=prediction, cpu_factorized_prediction=cpu_prediction, target=forces, **pre_arrays)
        result = {"fit_seconds": seconds, "diagnostics": diag, "convergence": flags,
                  "debias_seconds": debias_seconds,
                  "external_cpu_alpha_grid_seconds": grid_seconds,
                  "alpha_grid_backend": "cpu_derive_alpha_grid",
                  "automatic_alpha_grid": alpha,
                  "cv_settings": {"tol": args.cv_tol, "max_iter": args.cv_max_iter,
                      "matches_historical_defaults": args.cv_tol == 1e-3 and args.cv_max_iter == 800},
                  "alpha_auto_requested": True,
                  "cv_convergence_evidence": "exposed" if cv_flags else "unavailable",
                  "warnings": [str(w.message) for w in captured], "effective_environment": env,
                  "finite": bool(np.isfinite(coef).all() and np.isfinite(prediction).all()),
                  "cpu_prediction_relative_error": relative_error(prediction, cpu_prediction, np),
                  "force_relative_error": relative_error(prediction, forces, np),
                  "nonzero": int(np.count_nonzero(coef)),
                  "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
        result["stages"] = {
            "regularized_refit": diag["model"].get("regularized_solver_info_", "unavailable"),
            "debias": {"enabled": bool(args.debias), "iterative_solves": debias_records,
                "backend_diagnostics": diag["model"].get("debias_solver_info_", diag["results"].get("debias_solver_info")),
                "convergence_evidence": "exposed" if debias_records or diag["model"].get("debias_solver_info_") or diag["results"].get("debias_solver_info") else "unavailable_or_noniterative",
                "pre_force_relative_error": relative_error(pre_prediction, forces, np) if pre_coef is not None else None,
                "post_force_relative_error": relative_error(prediction, forces, np),
                "support_preserved": bool(np.all(coef[pre_coef == 0] == 0)) if pre_coef is not None else None}}
        if gpu:
            result["requested_active_devices"] = [str(selected_device) for selected_device in selected_devices]
            result["cuda_devices"] = [dict(device=str(selected_device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(selected_device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(selected_device))
                for selected_device in selected_devices]
            result["cuda_peak_allocated_bytes"] = result["cuda_devices"][0]["peak_allocated_bytes"]
            result["cuda_peak_reserved_bytes"] = result["cuda_devices"][0]["peak_reserved_bytes"]
        write_json(output / (label + "_result.json"), result)
        return result, coef, prediction


def main():
    p = parser()
    args = p.parse_args()
    if args.synthetic == (args.data is not None):
        p.error("provide exactly one DATA directory or --synthetic")
    if (args.group_size is not None and args.group_size < 1) or args.device < 0:
        p.error("group-size must be positive and device nonnegative")
    if args.cv_tol <= 0 or args.cv_max_iter < 1:
        p.error("CV tolerance and iteration limit must be positive")
    if args.prediction_rtol <= 0 or args.fit_rtol <= 0:
        p.error("comparison tolerances must be positive")
    # Atomic exclusive directory creation: also rejects existing empty dirs and
    # symlinks. Never use exist_ok, and do not write inside the input directory.
    output = args.output.absolute()
    if args.data and output.resolve().is_relative_to(args.data.resolve()):
        p.error("output must not be inside the read-only data directory")
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "FAIL", "limitations": [
        "CUDA peaks are PyTorch allocator peaks, not total device memory; RSS peak is process-lifetime.",
        "CPU factorized prediction validates application of fitted coefficients, not independent optimization.",
        "No throughput or speedup claim without representative real-cache hardware measurements.",
        "Automatic alpha grid is derived on CPU, included in fit timing; CUDA residency applies to the solver stage, not every pipeline operation.",
        "Strict CV defaults (1e-6/20000) differ from historical CV (1e-3/800); use --cv-tol 1e-3 --cv-max-iter 800 for a matched benchmark."]}
    start = time.monotonic()
    try:
        import numpy as np
        from scipy import sparse as sp
        try:
            import torch
        except ImportError:
            if not args.cpu_only:
                raise
            torch = None
        # This validator targets its own checkout, not an installed production package.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from core import optimizer as om
        expected_source = Path(__file__).resolve().parents[1] / "core" / "optimizer.py"
        if Path(om.__file__).resolve() != expected_source:
            raise RuntimeError("pheasy_gpu import resolves to another checkout: " + str(om.__file__))
        write_json(output / "configuration.json", {
            "arguments": vars(args), "optimizer_source": om.__file__, "numpy": np.__version__,
            "cv": {"group_size": args.group_size, "seed": args.seed,
                   "tol": args.cv_tol, "max_iter": args.cv_max_iter,
                   "historical_defaults": {"tol": 1e-3, "max_iter": 800},
                   "matches_historical_defaults": args.cv_tol == 1e-3 and args.cv_max_iter == 800,
                   "split": "random_rows" if not args.group_size or args.group_size == 1 else "grouped",
                   "interpretation": "Model-selection CV, not an independent physical holdout; reported force errors use the fitting data."},
            "torch": torch.__version__ if torch is not None else None, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "inherited_environment": {k: v for k, v in os.environ.items() if k.startswith("PHEASY_")},
            "baseline": {"nalpha": 20, "cv": 5, "tol": 1e-6, "max_iter": 20000, "standardize": True,
                         "debias": bool(args.debias), "alpha_auto": True, "decades": 4.0}})
        if not args.cpu_only:
            if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
                raise RuntimeError("requested logical CUDA device unavailable; no CPU fallback permitted")
            torch.cuda.set_device(args.device)
            result["device"] = {"logical_index": args.device, "name": torch.cuda.get_device_name(args.device)}
        prime, ns, forces, provenance = load_data(args, np, sp)
        write_json(output / "inputs.json", provenance)
        if prime.shape[1] != ns.shape[0] or prime.shape[0] != forces.size:
            raise ValueError("incompatible SM_prime/NS/force dimensions")
        group_size = args.group_size or 1
        if forces.size % group_size or forces.size // group_size < 5:
            raise ValueError("need at least five complete CV groups")
        if not all(np.isfinite(a).all() for a in (prime.data, ns.data, forces)):
            raise ValueError("non-finite input data")
        result["shape"] = [prime.shape[0], ns.shape[1]]
        primary, coef, prediction = run_fit(args, output, prime, ns, forces, np, torch, om, not args.cpu_only)
        result["primary"] = primary
        if not args.cpu_only:
            primary_index = torch.device(primary["requested_active_devices"][0]).index
            result["device"] = {"logical_index": primary_index,
                                "name": torch.cuda.get_device_name(primary_index)}
        failures = []
        if not primary["finite"] or primary["cpu_prediction_relative_error"] > args.prediction_rtol:
            failures.append("nonfinite result or CPU factorized prediction mismatch")
        if not primary["convergence"]:
            failures.append("convergence diagnostics unavailable (iterations alone are not proof)")
        if any(not f["converged"] for f in primary["convergence"]):
            failures.append("one or more exposed solves did not converge; full path was allowed to finish")
        if not args.cpu_only:
            # Exact dispatch assertion is intentionally conservative; an opt-in
            # environment variable or GPU availability alone proves nothing.
            observed = primary["diagnostics"]
            backend = observed["results"].get("execution_backend")
            refit_info = observed["model"].get("regularized_solver_info_", {})
            if backend != "gpu_twolevel_resident" or refit_info.get("backend") != backend:
                failures.append("resident dispatch not certified by execution/refit diagnostics: " + repr(backend))
            expected_devices = primary["requested_active_devices"]
            if observed["model"].get("cv_devices_") != expected_devices:
                failures.append("resident CV did not initialize the requested devices")
            fold_devices = observed["model"].get("cv_fold_devices_", [])
            if len(fold_devices) != 5 or any(device not in expected_devices for device in fold_devices):
                failures.append("missing or invalid actual fold device records")
            elif set(fold_devices) != set(expected_devices):
                failures.append("one or more requested GPUs executed no CV fold")
            if refit_info.get("device") != expected_devices[0]:
                failures.append("resident refit did not use requested logical CUDA device")
            actual_grid = np.asarray(observed["model"].get("alphas_", []))
            requested_grid = np.asarray(primary["automatic_alpha_grid"])
            if actual_grid.shape != requested_grid.shape or not np.allclose(actual_grid, requested_grid, rtol=1e-12, atol=0):
                failures.append("resident solver did not use the externally derived automatic grid")
            cv_records = observed["model"].get("cv_solver_info_")
            complete_cv = (isinstance(cv_records, list) and len(cv_records) == 5
                           and all(isinstance(fold, list) and len(fold) == 20
                                   and all(isinstance(solve, dict) and isinstance(solve.get("converged"), bool)
                                           for solve in fold) for fold in cv_records))
            if not complete_cv:
                failures.append("resident CV diagnostics must include all 5 folds x 20 alphas with explicit convergence flags")
        if args.cpu_reference and not args.cpu_only:
            reference, refcoef, refpred = run_fit(args, output, prime, ns, forces, np, torch, om, False)
            result["cpu_reference"] = reference
            result["comparison"] = {"coefficient_relative_error": relative_error(coef, refcoef, np),
                                    "prediction_relative_error": relative_error(prediction, refpred, np)}
            if result["comparison"]["prediction_relative_error"] > args.fit_rtol:
                failures.append("CPU/GPU independently fitted predictions differ")
            with np.load(output / "gpu_arrays.npz", allow_pickle=False) as ga, np.load(output / "cpu_arrays.npz", allow_pickle=False) as ca:
                if "pre_debias_prediction" in ga and "pre_debias_prediction" in ca:
                    result["comparison"]["pre_debias_prediction_relative_error"] = relative_error(ga["pre_debias_prediction"], ca["pre_debias_prediction"], np)
                    result["comparison"]["pre_debias_coefficient_relative_error"] = relative_error(ga["pre_debias_coefficients"], ca["pre_debias_coefficients"], np)
                    if result["comparison"]["pre_debias_prediction_relative_error"] > args.fit_rtol:
                        failures.append("CPU/GPU regularized predictions differ before debias")
            ga = np.asarray(primary["diagnostics"]["model"].get("alphas_", []))
            ca = np.asarray(reference["diagnostics"]["model"].get("alphas_", []))
            result["comparison"]["alpha_grid_matches"] = bool(ga.shape == ca.shape and ga.size == 20 and np.allclose(ga, ca, rtol=1e-8, atol=0))
            if not result["comparison"]["alpha_grid_matches"]:
                failures.append("CPU/GPU automatic alpha grids differ")
            if not reference["finite"] or any(not f["converged"] for f in reference["convergence"]):
                failures.append("CPU reference is nonfinite or has unconverged solves")
        result["failures"] = failures
        result["status"] = "FAIL" if failures else "CPU_SMOKE_PASS" if args.cpu_only else "PASS"
    except Exception:
        result["exception"] = traceback.format_exc()
    finally:
        result["total_seconds"] = time.monotonic() - start
        write_json(output / "result.json", result)
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
