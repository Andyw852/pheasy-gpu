#!/usr/bin/env python3
"""Small numerical CPU/single-GPU/multi-GPU validation (no material files).

Run on the GPU host: python dev/validate_gpu_backends.py (all visible devices)
                    or --devices 0,1 for an explicit selection.
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

# Support both installed package runs and this flat checkout.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))

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
    try:
        from pheasy_gpu.core import optimizer as om, gpu_backend as gb
    except ModuleNotFoundError as exc:
        if exc.name != "pheasy_gpu":
            raise
        from core import optimizer as om, gpu_backend as gb
    start = time.perf_counter()
    iterative_records = []

    def audit_fista(original, on_gpu):
        signature = inspect.signature(original)

        def checked(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if bound.arguments['_info'] is None:
                bound.arguments['_info'] = {}
            result = original(*bound.args, **bound.kwargs)
            count = int(result[1] if on_gpu else bound.arguments['_info']['n_iter'])
            limit = int(bound.arguments['max_iter'])
            iterative_records.append(dict(solver='GPU FISTA' if on_gpu else 'FISTA',
                                          itn=count, limit=limit, **bound.arguments['_info']))
            if not bound.arguments['_info'].get('converged', False) or count >= limit:
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
    if diagnostics is not None:
        if diagnostics.get("solver_kind") == "direct":
            if diagnostics.get("solver") != "TSQR" or diagnostics.get("rank_safe") is not True:
                raise AssertionError("unverified direct solver: " + repr(diagnostics))
        elif not diagnostics.get("converged", False):
            raise AssertionError("iterative solver did not converge: " + repr(diagnostics))
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


def _fista_boundary_check(gb, device):
    """Direct CUDA regressions: tiny steps, weighted KKT and zero-alpha metadata."""
    import torch
    records = []
    for diagonal, y, alpha, weights, limit, tol in [
            ([1., 1e-6], [0., 1e-6], 1e-16, None, 80, 1e-7),
            ([1., 2., 3.], [1., -2., .1], .01, [1., 2., 0.], 2000, 1e-9),
            ([1., 2., 3.], [1., -2., .1], 0., None, 2000, 1e-9)]:
        A = np.diag(diagonal)
        y = np.asarray(y)
        G = torch.as_tensor(A.T @ A, dtype=torch.float64, device="cuda:%d" % device)
        b = torch.as_tensor(A.T @ y, dtype=torch.float64, device=G.device)
        info = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            coef, count = gb._fista_gram(G, b, alpha, max_iter=limit, tol=tol,
                                        lipschitz=float(max(diagonal)**2),
                                        penalty_weights=weights, n_samples=len(y), _info=info)
        if alpha == 1e-16:
            assert count == limit and not info["converged"]
            assert info["kkt_relative"] > .9
            assert any("FISTA did not converge" in str(w.message) for w in caught)
        else:
            penalty = alpha * len(y) * np.asarray(weights if weights is not None else 1.)
            rhs = A.T @ y
            expected = np.sign(rhs) * np.maximum(np.abs(rhs) - penalty, 0.) / np.diag(A)**2
            np.testing.assert_allclose(coef.cpu().numpy(), expected, atol=1e-8, rtol=1e-8)
            assert info["converged"] and info["kkt_relative"] <= tol
        records.append(dict(alpha=alpha, **info))
    return records


def _debias_full_support_gpu(Optimizer, TwoLevelSM, devices):
    with _environment(PHEASY_GPU_SM=1, PHEASY_GPU_SM_DEVICES=",".join(map(str, devices)),
                      PHEASY_GPU_SM_NGPU=len(devices), PHEASY_SM_DTYPE="float64"):
        diagonal = np.arange(1., 25.)
        op = TwoLevelSM(sp.diags(diagonal, format="csr"), sp.eye(24, format="csr"), dtype=np.float64)
        try:
            assert op._gpu_mv is not None and op._gpu_mv._devs == devices
            before = getattr(op._gpu_mv, "_n_calls", 0)
            expected = np.linspace(.2, 1.2, 24)
            model = Optimizer.__new__(Optimizer)
            actual = model._debias(op, diagonal * expected, expected * .5)
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
            assert op._gpu_mv is not None and op._gpu_mv._devs == devices
            calls = op._gpu_mv._n_calls - before
            assert calls > 0
            return dict(devices=devices, gpu_spmv_calls=calls, max_abs_error=float(np.max(np.abs(actual-expected))))
        finally:
            if op._gpu_mv is not None:
                op._gpu_mv.close()


def _resident_rfe_check(Optimizer, matrix, forces):
    models = []
    for resident in (0, 1):
        with _environment(PHEASY_GPU_RFE_RESIDENT=resident, PHEASY_RFE_JACOBI=0, PHEASY_GPU_RFE_RANKING=0):
            model = Optimizer("RFE", cv=3, rand_seed=17, use_gpu=True)
            model.fit(matrix, forces)
            models.append(model)
    ref, got = models
    meta = got.results["backend_metadata"]
    if not meta.get("resident_subset_inputs") or meta.get("cv_fold_scoring") != "gpu":
        raise AssertionError("resident RFE was not dispatched: " + repr(meta))
    if meta.get("resident_input_kind") in ("csr", "twolevel"):
        diagnostics = meta.get("iterative_diagnostics", [])
        if not diagnostics or len(diagnostics) != meta.get("gpu_subset_solves"):
            raise AssertionError("missing per-subset iterative diagnostics")
        if got.results.get("execution_backend") != "gpu_rfe_resident_iterative":
            raise AssertionError("incorrect public iterative backend label")
        if not all(d.get("converged") and str(d.get("device", "")).startswith("cuda") for d in diagnostics):
            raise AssertionError("unconverged or non-CUDA subset solve")
        if {d.get("fit_scope") for d in diagnostics} != {"full", "fold"}:
            raise AssertionError("expected both full and CV fold diagnostics")
    np.testing.assert_allclose(got.predict(matrix), ref.predict(matrix), atol=1e-9, rtol=1e-9)
    np.testing.assert_array_equal(got._model.support_, ref._model.support_)
    np.testing.assert_allclose(got._model.best_rmse_cv_, ref._model.best_rmse_cv_, atol=1e-10, rtol=1e-9)
    return dict(metadata=meta, relative_prediction_error=_relative(got.predict(matrix), ref.predict(matrix)))


def benchmark_ridge_rfe(Optimizer, torch, repeats=3, rows=1200, columns=96, input_kind="dense", ns_kind="identity", TwoLevelSM=None):
    """Matched warm end-to-end fits; data creation and CUDA warmup excluded."""
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(2026)
    if rows < 18 or rows % 6 or columns < 12 or repeats < 1:
        raise ValueError("benchmark requires rows >= 18 divisible by 6, columns >= 12, repeats >= 1")
    if input_kind == "dense":
        A = rng.normal(size=(rows, columns))
    else:
        import scipy.sparse as sp
        A = sp.random(rows, columns, density=.1, format="csr", random_state=rng, data_rvs=rng.standard_normal)
        if input_kind == "twolevel":
            # Build the operator from the SAME class object that Optimizer uses.
            # Importing it independently (e.g. `core.optimizer` while Optimizer came
            # from `pheasy_gpu.core.optimizer`) yields a distinct class, so the
            # resident gate's isinstance check rejects it and silently falls back to
            # CPU -- the benchmark then measures the wrong path.
            if TwoLevelSM is None:
                raise ValueError("twolevel benchmark requires the Optimizer module's TwoLevelSM")
            ns = sp.eye(columns, format="csr")
            if ns_kind == "mixed":
                ns = ns + .1 * sp.random(columns, columns, density=.1, format="csr", random_state=rng, data_rvs=rng.standard_normal)
            A = TwoLevelSM(A, ns)
    truth = np.zeros(columns)
    truth[:12] = rng.normal(size=12)
    y = A @ truth + rng.normal(scale=.01, size=rows)
    report = {"input_kind": input_kind, "shape": list(A.shape), "repeats": repeats, "cpu_threads": 2,
              "device": torch.cuda.get_device_name(torch.cuda.current_device()), "torch": torch.__version__,
              "timing": "Optimizer construction + fit including transfer and metrics; warmup excluded", "runs": []}
    report["fixture"] = {"density": 1.0 if input_kind == "dense" else .1,
                         "twolevel_ns": ns_kind if input_kind == "twolevel" else None}
    report["configuration_note"] = "gpu is the requested flag; backend and metadata identify actual execution"
    torch.ones(1, device="cuda").sum().item()
    benchmark_env = dict(PHEASY_N_JOBS=1, PHEASY_RFE_N_JOBS=1, PHEASY_CV_GROUP_SIZE=6,
                         PHEASY_RFE_STEP=.5, PHEASY_RFE_MIN_FEATURES=6, PHEASY_RFE_PATIENCE=3,
                         PHEASY_RFE_RIDGE_ALPHA=0, PHEASY_RFE_JACOBI=0)
    report["fixed_environment"] = benchmark_env
    report["seeds"] = {"data": 2026, "cv": 17}
    report["cv_folds"] = 3
    report["ridge_alphas"] = np.logspace(-6, -2, 5).tolist()
    with threadpool_limits(limits=2), _environment(**benchmark_env):
        for method in (("RIDGE", "RFE") if input_kind == "dense" else ("RFE",)):
            reference = None
            for gpu, ranking, resident in ((False, 0, 0), (True, 0, 0), (True, 1, 0), (True, 0, 1), (True, 1, 1)):
                if method == "RIDGE" and (ranking or resident):
                    continue
                times = []
                with _environment(PHEASY_GPU_RFE_RANKING=ranking, PHEASY_GPU_RFE_RESIDENT=resident):
                    for repeat in range(repeats + 1):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        model = Optimizer(method, alpha=np.logspace(-6, -2, 5), cv=3, rand_seed=17, standardize=method == "RIDGE", use_gpu=gpu)
                        model.fit(A, y)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        metadata = model.results.get("backend_metadata") or {}
                        if resident and (not metadata.get("resident_subset_inputs") or metadata.get("cv_fold_scoring") != "gpu"):
                            raise AssertionError("benchmark resident dispatch failed: " + repr(metadata))
                        if resident and ranking and not metadata.get("gpu_importance_rounds", 0):
                            raise AssertionError("benchmark did not execute CUDA importance calculation")
                        if repeat:
                            times.append(elapsed)
                prediction = model.predict(A)
                if reference is None:
                    reference = prediction
                difference = float(np.linalg.norm(prediction-reference) / max(np.linalg.norm(reference), 1e-30))
                if not np.isfinite(difference) or difference > 1e-7:
                    raise AssertionError(f"{method} benchmark prediction parity failed: {difference}")
                report["runs"].append(dict(method=method, gpu=gpu, ranking=ranking, resident=resident, seconds=times, median_seconds=float(np.median(times)), prediction_relative_difference=difference, backend=model.results.get("execution_backend"), metadata=model.results.get("backend_metadata")))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default=None,
                        help="visible CUDA device indices; default is all visible devices (at most six)")
    parser.add_argument("--json", default="gpu_backend_validation.json")
    parser.add_argument("--methods", nargs="+",
                        default=["OLS", "RIDGE", "LASSO", "ALASSO", "RFE", "RFE-OLS-TSQR"])
    parser.add_argument("--benchmark-ridge-rfe", action="store_true", help="matched repeated local end-to-end timing instead of acceptance suite")
    parser.add_argument("--benchmark-rows", type=int, default=1200)
    parser.add_argument("--benchmark-columns", type=int, default=96)
    parser.add_argument("--benchmark-input", choices=("dense", "csr", "twolevel"), default="dense")
    parser.add_argument("--benchmark-ns", choices=("identity", "mixed"), default="identity", help="TwoLevel NS: identity or identity plus sparse random mixing")
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    args = parser.parse_args()
    import torch
    if args.devices is None:
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = list(dict.fromkeys(int(value) for value in args.devices.split(",")))
    if not 1 <= len(devices) <= 6:
        parser.error("select between one and six distinct devices")
    try:
        from pheasy_gpu.core import gpu_backend as gb
        from pheasy_gpu.core.optimizer import Optimizer, TwoLevelSM
    except ModuleNotFoundError as exc:
        if exc.name != "pheasy_gpu":
            raise
        from core import gpu_backend as gb
        from core.optimizer import Optimizer, TwoLevelSM
    if not torch.cuda.is_available() or any(d < 0 or d >= torch.cuda.device_count() for d in devices):
        parser.error("requested CUDA devices are unavailable")
    if args.benchmark_ridge_rfe:
        with _environment(PHEASY_GPU_DEVICE=devices[0]):
            torch.cuda.set_device(devices[0])
            report = benchmark_ridge_rfe(Optimizer, torch, args.benchmark_repeats, args.benchmark_rows, args.benchmark_columns, args.benchmark_input, args.benchmark_ns, TwoLevelSM)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
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
        check("fista/direct_cuda_boundaries", lambda: _fista_boundary_check(gb, devices[0]))
        check("debias/full_support_gpu", lambda: _debias_full_support_gpu(Optimizer, TwoLevelSM, devices))
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
            if "RFE" in args.methods:
                check("resident_rfe/%s" % ("rankdef" if rank_deficient else "wellconditioned"),
                      lambda: _resident_rfe_check(Optimizer, fixture, forces))
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
            if "RFE" in args.methods:
                condition = "rankdef" if rank_deficient else "wellconditioned"
                check("resident_rfe/csr/" + condition, lambda: _resident_rfe_check(Optimizer, prime @ ns, forces))
                check("resident_rfe/twolevel/" + condition, lambda: _resident_rfe_check(Optimizer, TwoLevelSM(prime, ns), forces))
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
