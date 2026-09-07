#!/usr/bin/env python3
"""Small numerical CPU/single-GPU/multi-GPU validation (no material files).

Run on the GPU host: python dev/validate_gpu_backends.py --devices 0,1,2,3,4,5
All six methods run on well-conditioned and rank-deficient dense fixtures.
The large-memory path is checked through TwoLevelSM matvec/adjoint operations
and all six methods on one and all requested GPUs. JSON contains every
check, dispatch counters, numerical errors, timings and device metadata.
"""
import argparse
import contextlib
import json
import inspect
import os
from pathlib import Path
import time
import traceback
import warnings
from unittest.mock import patch

import numpy as np
from scipy import sparse as sp
from sklearn.exceptions import ConvergenceWarning


@contextlib.contextmanager
def _environment(**values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _dispatch_counts(gb):
    counts = {}

    def instrument(owner, name, label):
        original = getattr(owner, name)

        def counted(*args, **kwargs):
            counts[label] = counts.get(label, 0) + 1
            try:
                result = original(*args, **kwargs)
                if name == '_to_torch':
                    if result.device.type != 'cuda':
                        raise AssertionError('GPU dense backend created a CPU tensor')
                    device_label = 'tensor_device/' + str(result.device)
                    counts[device_label] = counts.get(device_label, 0) + 1
                return result
            except Exception:
                counts[label + '.errors'] = counts.get(label + '.errors', 0) + 1
                raise

        return patch.object(owner, name, counted)

    with contextlib.ExitStack() as stack:
        for name in ("lstsq", "qr_solve", "ridge_solve", "_to_torch"):
            stack.enter_context(instrument(gb, name, name))
        for name in ("GpuLassoCV", "GpuRidgeCV"):
            stack.enter_context(instrument(getattr(gb, name), "fit", name + ".fit"))
        yield counts


def _relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _fit(Optimizer, method, matrix, forces, gpu):
    from pheasy_gpu.core import optimizer as om, gpu_backend as gb
    start = time.perf_counter()
    iterative_records = []

    def audit_fista(original, on_gpu):
        signature = inspect.signature(original)

        def checked(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if not on_gpu and bound.arguments['_info'] is None:
                bound.arguments['_info'] = {}
            result = original(*bound.args, **bound.kwargs)
            count = int(result[1] if on_gpu else bound.arguments['_info']['n_iter'])
            limit = int(bound.arguments['max_iter'])
            iterative_records.append(dict(solver='GPU FISTA' if on_gpu else 'FISTA',
                                          itn=count, limit=limit, converged=count < limit))
            if count >= limit:
                raise AssertionError('FISTA reached iteration limit: %d' % count)
            return result
        return checked

    optimizer = Optimizer(
        method, alpha=np.logspace(-6, -2, 5), nalpha=5, cv=3,
        alpha_auto=False, tol=1e-8, max_iter=5000, rand_seed=17,
        standardize=method in ("LASSO", "ALASSO", "RIDGE"), use_gpu=gpu,
    )
    with warnings.catch_warnings(), \
            patch.object(om, '_fista_lasso', audit_fista(om._fista_lasso, False)), \
            patch.object(gb, '_fista_gram', audit_fista(gb._fista_gram, True)):
        warnings.simplefilter('error', ConvergenceWarning)
        warnings.filterwarnings('error', message=r'LSM[QR] did not converge:.*',
                                category=RuntimeWarning)
        warnings.filterwarnings('error', message=r'LSQR did not converge:.*',
                                category=RuntimeWarning)
        optimizer.fit(matrix, forces)
    if any(not item['converged'] for item in iterative_records):
        raise AssertionError('an unconverged FISTA call was caught by a fallback')
    diagnostics = optimizer.results.get('solver_info')
    if diagnostics is not None and not diagnostics['converged']:
        raise AssertionError('iterative solver did not converge: ' + repr(diagnostics))
    coefficients = np.asarray(optimizer.results["coef"])
    if not np.isfinite(coefficients).all():
        raise AssertionError("non-finite coefficients")
    return coefficients, {
        "seconds": time.perf_counter() - start,
        "relative_residual": float(optimizer.metrics["re"]),
        "alpha": optimizer.results.get("alpha"),
        "nonzero": int(np.count_nonzero(coefficients)),
        "solver_info": optimizer.results.get("solver_info"),
        "fista_records": iterative_records,
    }


def _dense_check(Optimizer, gb, method, matrix, forces, rank_deficient):
    cpu, cpu_info = _fit(Optimizer, method, matrix, forces, False)
    with _dispatch_counts(gb) as counts:
        gpu, gpu_info = _fit(Optimizer, method, matrix, forces, True)
    if not any(key.startswith('tensor_device/cuda:') for key in counts):
        raise AssertionError("GPU fit created no verified CUDA tensor")
    if any(key.endswith('.errors') for key in counts):
        raise AssertionError('GPU dense solver error or hidden fallback: ' + repr(counts))
    prediction_error = _relative(matrix @ gpu, matrix @ cpu)
    coefficient_error = _relative(gpu, cpu)
    info = dict(cpu=cpu_info, gpu=gpu_info, gpu_dispatch=counts,
                relative_prediction_error=prediction_error,
                relative_coefficient_error=coefficient_error)
    # Rank-deficient L1/RFE coefficients are not unique; compare predictions.
    tolerance = 2e-3 if rank_deficient else 2e-4
    if prediction_error > tolerance:
        raise AssertionError("CPU/GPU predictions differ: " + json.dumps(info))
    if method == "OLS" and coefficient_error > 1e-7:
        raise AssertionError("SVD minimum-norm coefficients differ: " + json.dumps(info))
    if gpu_info["relative_residual"] > 0.02:
        raise AssertionError("synthetic recovery residual exceeds 2%: " + json.dumps(info))
    return info


def _twolevel_check(Optimizer, TwoLevelSM, prime, nullspace, forces, devices, stress_pairs=500):
    explicit = (prime @ nullspace).toarray()
    rng = np.random.default_rng(81)
    baseline = {}
    with _environment(PHEASY_GPU_SM=0):
        cpu_operator = TwoLevelSM(prime, nullspace, dtype=np.float64)
        for method in ("OLS", "RIDGE", "LASSO", "ALASSO", "RFE", "RFE-OLS-TSQR"):
            baseline[method] = _fit(Optimizer, method, cpu_operator, forces, False)
    with _environment(PHEASY_GPU_SM=1, PHEASY_GPU_SM_NGPU=len(devices),
                      PHEASY_GPU_SM_DEVICES=",".join(map(str, devices))):
        operator = TwoLevelSM(prime, nullspace, dtype=np.float64)
        if operator._gpu_mv is None or operator._gpu_mv._devs != devices:
            raise AssertionError("GPU-SM unexpectedly fell back or selected other devices")
        try:
            mv_error = adjoint_error = 0.0
            for _ in range(stress_pairs):
                x = rng.normal(size=explicit.shape[1])
                u = rng.normal(size=explicit.shape[0])
                mv_error = max(mv_error, _relative(operator @ x, explicit @ x))
                adjoint_error = max(adjoint_error, _relative(operator.rmatvec(u), explicit.T @ u))
            if max(mv_error, adjoint_error) > 1e-12:
                raise AssertionError("GPU sparse matvec/transpose differs from CPU")
            fits = {}
            for method in ("OLS", "RIDGE", "LASSO", "ALASSO", "RFE", "RFE-OLS-TSQR"):
                before = operator._gpu_mv._n_calls
                gpu, info = _fit(Optimizer, method, operator, forces, True)
                cpu, cpu_info = baseline[method]
                error = _relative(explicit @ gpu, explicit @ cpu)
                if error > 2e-5:
                    raise AssertionError("%s two-level prediction mismatch %.3e" % (method, error))
                if operator._gpu_mv is None:
                    raise AssertionError("GPU-SM failed during fit and silently fell back")
                calls = operator._gpu_mv._n_calls - before
                if calls <= 0:
                    raise AssertionError('fit executed no GPU SpMV')
                fits[method] = dict(cpu=cpu_info, gpu=info, relative_prediction_error=error,
                                    gpu_spmv_calls=calls)
            return dict(devices=devices, stress_matvec_pairs=stress_pairs, relative_matvec_error=mv_error,
                        relative_transpose_error=adjoint_error, fits=fits)
        finally:
            if operator._gpu_mv is not None:
                operator._gpu_mv.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3,4,5",
                        help="visible CUDA device indices; at most six")
    parser.add_argument("--json", default="gpu_backend_validation.json")
    parser.add_argument("--methods", nargs="+",
                        default=["OLS", "RIDGE", "LASSO", "ALASSO", "RFE", "RFE-OLS-TSQR"])
    args = parser.parse_args()
    devices = list(dict.fromkeys(int(value) for value in args.devices.split(",")))
    if not 1 <= len(devices) <= 6:
        parser.error("select between one and six distinct devices")
    import torch
    from pheasy_gpu.core import gpu_backend as gb
    from pheasy_gpu.core.optimizer import Optimizer, TwoLevelSM
    if not torch.cuda.is_available() or any(d < 0 or d >= torch.cuda.device_count() for d in devices):
        parser.error("requested CUDA devices are unavailable")
    report = dict(
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        devices=[dict(index=d, name=torch.cuda.get_device_name(d)) for d in devices], checks=[],
    )

    def check(name, action):
        print("[VALIDATE] " + name, flush=True)
        start = time.perf_counter()
        try:
            details = action()
            record = dict(name=name, status="PASS", details=details)
        except Exception:
            record = dict(name=name, status="FAIL", traceback=traceback.format_exc())
        record["seconds"] = time.perf_counter() - start
        report["checks"].append(record)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[VALIDATE] %s %s %.2fs" % (name, record["status"], record["seconds"]), flush=True)
        if record["status"] == "FAIL":
            print(record["traceback"], flush=True)

    with _environment(
        PHEASY_GPU_DEVICE=devices[0], PHEASY_GPU_DEVICES=devices[0], PHEASY_GPU_SM=0,
        PHEASY_N_JOBS=1, PHEASY_RFE_N_JOBS=1, PHEASY_MAX_CORES=2,
        PHEASY_CV_GROUP_SIZE=6, PHEASY_RFE_STEP=0.5, PHEASY_RFE_MIN_FEATURES=3,
        PHEASY_TSQR_STEP=0.5, PHEASY_TSQR_MIN_FEATURES=3, PHEASY_RFE_PATIENCE=2,
        PHEASY_TSQR_CRITERION="cv", PHEASY_LASSO_DEBIAS=0, PHEASY_GPU_LASSO=1,
        PHEASY_TWOLEVEL_CACHE_T=0, PHEASY_OLS_ATOL=1e-10, PHEASY_OLS_BTOL=1e-10,
        PHEASY_OLS_MAXITER=2000, PHEASY_LSQR_ATOL=1e-10, PHEASY_LSQR_BTOL=1e-10,
        PHEASY_GRAM_MAX_GB=0, PHEASY_CV_TOL=1e-8, PHEASY_CV_MAX_ITER=5000,
    ):
        rng = np.random.default_rng(41)
        matrix = rng.normal(size=(180, 18))
        true_coef = np.zeros(18)
        true_coef[:4] = [1.0, -0.8, 0.4, 0.2]
        for rank_deficient in (False, True):
            fixture = matrix.copy()
            if rank_deficient:
                fixture[:, -1] = fixture[:, 0]
                fixture[:, -2] = fixture[:, 1] + fixture[:, 2]
            forces = fixture @ true_coef + rng.normal(scale=1e-4, size=180)
            for method in args.methods:
                check("dense/%s/%s" % ("rankdef" if rank_deficient else "wellconditioned", method),
                      lambda: _dense_check(Optimizer, gb, method, fixture, forces, rank_deficient))
        prime = sp.random(360, 36, density=0.35, random_state=rng,
                          data_rvs=lambda n: rng.normal(size=n), format="csr")
        nullspace = sp.vstack([sp.eye(24), sp.csr_matrix(rng.normal(scale=0.1, size=(12, 24)))], format="csr")
        true_coef = np.zeros(24)
        true_coef[:4] = [0.9, -0.7, 0.3, 0.1]
        for rank_deficient in (False, True):
            ns = nullspace.copy().tolil()
            if rank_deficient:
                ns[:, -1] = ns[:, 0]
            ns = ns.tocsr()
            forces = prime @ (ns @ true_coef) + rng.normal(scale=1e-4, size=360)
            for selected in ([devices[0]], devices):
                if len(selected) == 1 and selected is devices:
                    continue
                check("twolevel/%s/%d_gpu" % ('rankdef' if rank_deficient else 'wellconditioned', len(selected)),
                      lambda: _twolevel_check(Optimizer, TwoLevelSM, prime, ns, forces, selected))
    failures = sum(check["status"] == "FAIL" for check in report["checks"])
    report["summary"] = dict(passed=len(report["checks"]) - failures, failed=failures)
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"]), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
