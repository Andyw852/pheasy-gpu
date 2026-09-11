#!/usr/bin/env python3
"""Resident LASSO contract tests; CUDA availability is never simulated silently."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import optimizer as opt
from core import gpu_backend as gb

class ResidentDispatchTests(unittest.TestCase):
    def test_canonical_flag_selects_resident_without_editable_install(self):
        A = opt.TwoLevelSM(sp.eye(4, format="csr"), sp.eye(4, format="csr"))
        with patch.dict(os.environ, {"PHEASY_GPU_LASSO_RESIDENT": "1"}):
            self.assertEqual(opt._lasso_backend(A), "gpu_resident")
        with patch.dict(os.environ, {"PHEASY_GPU_LASSO_RESIDENT": "0", "PHEASY_GPU_TWOLEVEL_LASSO": "1"}):
            self.assertEqual(opt._lasso_backend(A), "iterative")

    def test_scaled_twolevel_preserves_resident_dispatch(self):
        A = opt.TwoLevelSM(sp.eye(4, format="csr"), sp.eye(4, format="csr"))
        scaled = opt._scale_columns(A, np.array([1., 2., 3., 4.]))
        with patch.dict(os.environ, {"PHEASY_GPU_LASSO_RESIDENT": "1"}):
            self.assertEqual(opt._lasso_backend(scaled), "gpu_resident")
            sliced = opt._row_slice(scaled, np.array([3, 1]))
            self.assertEqual(opt._lasso_backend(sliced), "gpu_resident")
            np.testing.assert_allclose(sliced @ np.ones(4), [.25, .5])

    def test_public_resident_alasso_uses_gpu_pilot_and_weighted_fista(self):
        if torch is None or not torch.cuda.is_available():
            self.skipTest("CUDA hardware required for resident ALASSO")
        A = opt.TwoLevelSM(sp.eye(4, format="csr"), sp.eye(4, format="csr"))
        model = opt.Optimizer("ALASSO", alpha=[.01, .1], nalpha=2, cv=2,
                              tol=1e-7, max_iter=200, use_gpu=True,
                              alpha_auto=True, standardize=False)
        with patch.dict(os.environ, {"PHEASY_GPU_LASSO_RESIDENT": "1",
                                     "PHEASY_LASSO_DEBIAS": "0",
                                     "PHEASY_CV_TOL": "1e-7",
                                     "PHEASY_CV_MAX_ITER": "200"}):
            model.fit(A, np.array([1., 0., 0., 0.]))
        self.assertEqual(model.results["execution_backend"], "gpu_twolevel_resident")
        self.assertEqual(model.results["regularized_solver_info"]["device"], "cuda:0")
        self.assertTrue(model._model.penalty_weights_ is not None)

    def test_explicit_resident_request_does_not_fall_back_to_cpu(self):
        A = opt.TwoLevelSM(sp.eye(8, format="csr"), sp.eye(8, format="csr"))
        with patch.dict(os.environ, {"PHEASY_GPU_LASSO_RESIDENT": "1", "PHEASY_GPU_SM": "0", "PHEASY_LASSO_DEBIAS": "0"}):
            model = opt.Optimizer("lasso", alpha=[.1], cv=2, use_gpu=False)
            with self.assertRaisesRegex(RuntimeError, "CUDA|GPU|gpu"):
                model.fit(A, np.arange(8.))

try:
    import torch
except ImportError:
    torch = None

@unittest.skipIf(torch is None, "Torch not installed")
class ResidentNumericsTests(unittest.TestCase):
    def test_resident_normalized_grouped_cv_matches_sklearn(self):
        from sklearn.linear_model import Lasso
        rng = np.random.default_rng(12)
        prime = sp.csr_matrix(rng.normal(size=(30, 9)))
        ns = sp.csr_matrix(rng.normal(size=(9, 5)))
        dense = (prime @ ns).toarray()
        y = dense @ np.array([1., -.7, 0., 0., .3])
        scale = np.linalg.norm(dense, axis=0)
        X = dense / scale
        alphas = [.001, .02, .1]
        splits = opt._make_cv_splits(30, 3, 42, 3)
        expected = np.zeros((3, 3))
        for i, alpha in enumerate(alphas):
            for k, (tr, va) in enumerate(splits):
                ref = Lasso(alpha=alpha, fit_intercept=False, tol=1e-12, max_iter=20000).fit(X[tr], y[tr])
                expected[i, k] = np.mean((X[va] @ ref.coef_ - y[va])**2)
        # Explicit CPU Torch numerical emulation, not GPU evidence.
        with patch.object(gb, "available", return_value=True), patch.object(gb, "enabled", return_value=True), patch.object(gb, "device", return_value="cpu"), patch.dict(os.environ, {"PHEASY_CV_TOL": "1e-9", "PHEASY_CV_MAX_ITER": "3000"}):
            model = gb.GpuTwoLevelLassoCV(alphas, 3, 1e-9, 3000, 42, group_size=3, standardize=True, alpha_auto=False)
            model.fit(opt.TwoLevelSM(prime, ns), y)
        np.testing.assert_allclose(model.mse_path_, expected, rtol=1e-6, atol=1e-8)
        self.assertEqual(model.alpha_, alphas[np.argmin(expected.mean(axis=1))])
        ref = Lasso(alpha=model.alpha_, fit_intercept=False, tol=1e-12, max_iter=20000).fit(X, y)
        np.testing.assert_allclose(model.coef_, ref.coef_ / scale, atol=1e-7)
        self.assertTrue(model.regularized_solver_info_["converged"])

    def test_optimizer_standardization_uses_resident_math_and_physical_coefficients(self):
        dense = np.diag([1., 2., 3., 4.])
        A = opt.TwoLevelSM(sp.csr_matrix(dense), sp.eye(4, format="csr"))
        with patch.object(gb, "available", return_value=True), patch.object(gb, "enabled", return_value=True), patch.object(gb, "device", return_value="cpu"), patch.dict(os.environ, {"PHEASY_GPU_LASSO_RESIDENT": "1", "PHEASY_LASSO_DEBIAS": "0"}), patch.object(A, "col_norms", side_effect=AssertionError("host normalization forbidden")):
            model = opt.Optimizer("lasso", alpha=[.1], cv=2, standardize=True, tol=1e-10, alpha_auto=False)
            model.fit(A, np.array([2., -2., .1, 0.]))
        # Unit normalized design is identity; soft threshold is n*alpha=.4.
        np.testing.assert_allclose(model.results["coef"], [1.6, -.8, 0., 0.], atol=1e-10)
        self.assertEqual(model.results["execution_backend"], "gpu_twolevel_resident")
        self.assertEqual(model.results["postfit_backend"], "cpu_metrics")

    def test_partial_allocation_failure_releases_owned_factors(self):
        A = opt.TwoLevelSM(sp.eye(4, format="csr"), sp.eye(4, format="csr"))
        operator = gb.GpuTwoLevelOperator.__new__(gb.GpuTwoLevelOperator)
        original = torch.sparse_csr_tensor
        calls = []
        def fail_second(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("simulated sparse allocation failure")
            return original(*args, **kwargs)
        with patch.object(gb, "available", return_value=True), patch.object(gb, "enabled", return_value=True), patch.object(gb, "device", return_value="cpu"), patch.object(torch, "sparse_csr_tensor", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                operator.__init__(A)
        self.assertIsNone(operator.prime)
        operator.close()

    def test_resident_default_debias_preserves_cpu_postfit_semantics(self):
        A = opt.TwoLevelSM(sp.eye(4, format="csr"), sp.eye(4, format="csr"))
        with patch.dict(os.environ, {"PHEASY_GPU_LASSO_RESIDENT": "1", "PHEASY_LASSO_DEBIAS": "1"}), patch.object(gb, "available", return_value=True), patch.object(gb, "enabled", return_value=True), patch.object(gb, "device", return_value="cpu"):
            model = opt.Optimizer("lasso", alpha=[.1], cv=2, tol=1e-10, alpha_auto=False)
            model.fit(A, np.array([2., -2., .1, 0.]))
        self.assertEqual(model.results["debias_backend"], "cpu_lsmr")
        self.assertEqual(model.results["postfit_backend"], "cpu_lsmr_and_cpu_metrics")
        np.testing.assert_allclose(model.results["pre_debias_coef"], [1.6, -1.6, 0., 0.], atol=1e-10)
        np.testing.assert_allclose(model.results["coef"], [2., -2., 0., 0.], atol=1e-8)

    def test_resident_lasso_auto_grid_matches_derive_alpha_grid(self):
        rng = np.random.default_rng(5)
        prime = sp.csr_matrix(rng.normal(size=(12, 5)))
        ns = sp.csr_matrix(rng.normal(size=(5, 4)))
        A = opt.TwoLevelSM(prime, ns)
        y = np.asarray(A @ np.linspace(-1, 1, 4)).ravel()
        nalpha, decades = 5, 4.0
        expected = opt.derive_alpha_grid(A, y, nalpha=nalpha, decades=decades, standardize=True)
        with patch.object(gb, "available", return_value=True), patch.object(gb, "enabled", return_value=True), patch.object(gb, "device", return_value="cpu"), patch.dict(os.environ, {"PHEASY_CV_TOL": "1e-6", "PHEASY_CV_MAX_ITER": "500"}):
            model = gb.GpuTwoLevelLassoCV([1.0], 3, 1e-6, 500, 0, standardize=True,
                                          nalpha=nalpha, decades=decades, alpha_auto=True)
            model.fit(A, y)
        # The resident KKT grid must reproduce derive_alpha_grid(standardize=True).
        np.testing.assert_allclose(model.alphas, expected, rtol=1e-12, atol=1e-14)

    def test_resident_debias_runs_gpu_cgls_on_cuda(self):
        if torch is None or not torch.cuda.is_available():
            self.skipTest("CUDA hardware required for resident debias")
        A = opt.TwoLevelSM(sp.eye(6, format="csr"), sp.eye(6, format="csr"))
        y = np.array([2., -2., .1, 0., .3, 0.])
        model = opt.Optimizer("lasso", alpha=[.1], cv=2, tol=1e-7, max_iter=300,
                              use_gpu=True, standardize=False, alpha_auto=False)
        with patch.dict(os.environ, {"PHEASY_GPU_LASSO_RESIDENT": "1", "PHEASY_LASSO_DEBIAS": "1"}):
            model.fit(A, y)
        self.assertEqual(model.results["execution_backend"], "gpu_twolevel_resident")
        self.assertEqual(model.results["debias_backend"], "gpu_cgls")
        self.assertEqual(model.results["postfit_backend"], "gpu_cgls_and_cpu_metrics")

    def test_memory_preflight_refuses_before_upload(self):
        A = opt.TwoLevelSM(sp.eye(4, format="csr"), sp.eye(4, format="csr"))
        with patch.object(gb, "available", return_value=True), patch.object(gb, "enabled", return_value=True), patch.object(gb, "device", return_value="cpu"), patch.object(gb, "_device_free_bytes", return_value=1), patch.object(torch, "as_tensor", side_effect=AssertionError("upload before preflight")):
            with self.assertRaisesRegex(MemoryError, "resident|Resident"):
                gb.GpuTwoLevelOperator(A)

    def test_nested_scaled_input_keeps_its_coordinate_system(self):
        A = opt.TwoLevelSM(sp.eye(4, format="csr"), sp.eye(4, format="csr"))
        A = opt._scale_columns(opt._scale_columns(A, np.array([1., 2., 3., 4.])), np.array([2., 1., 2., 1.]))
        with patch.object(gb, "available", return_value=True), patch.object(gb, "enabled", return_value=True), patch.object(gb, "device", return_value="cpu"):
            model = gb.GpuTwoLevelLassoCV([.1], 2, 1e-10, 1000, 0, standardize=True, alpha_auto=False).fit(A, np.array([2., -2., .1, 0.]))
        np.testing.assert_allclose(model.coef_, [3.2, -3.2, 0., 0.], atol=1e-10)

    def test_sparse_factor_adjoint_norms_and_no_numpy_iteration_transfers(self):
        prime = sp.csr_matrix([[1., 2., 0.], [0., -1., 3.], [2., 0., 1.]])
        ns = sp.csr_matrix([[2., 0., 0.], [1., -1., 0.], [0., 2., 0.]])
        dense = (prime @ ns).toarray()
        with patch.object(gb, "available", return_value=True), patch.object(gb, "enabled", return_value=True), patch.object(gb, "device", return_value="cpu"):
            operator = gb.GpuTwoLevelOperator(opt.TwoLevelSM(prime, ns))
        v = torch.tensor([1., -.5, 3.], dtype=torch.float64)
        u = torch.tensor([2., 3., -1.], dtype=torch.float64)
        np.testing.assert_allclose(operator.matvec(v).numpy(), dense @ v.numpy())
        np.testing.assert_allclose(operator.rmatvec(u).numpy(), dense.T @ u.numpy())
        norms = operator.normalize()
        np.testing.assert_allclose(norms.numpy(), [np.sqrt(33.), np.sqrt(57.), 1.])
        # Guard public Tensor host-vector conversion APIs across the solver.
        with patch.object(torch.Tensor, "numpy", side_effect=AssertionError("numpy transfer")), patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU transfer")):
            coef, info = gb._fista_twolevel(operator, u, .1, None, 1000, 1e-9, operator.lipschitz())
        self.assertTrue(info["converged"])
        self.assertEqual(coef.device.type, "cpu")  # explicitly emulated, not CUDA

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA hardware required")
    def test_real_cuda_residency_and_analytical_solution(self):
        with patch.object(gb, "enabled", return_value=True):
            A = opt.TwoLevelSM(sp.eye(4, format="csr"), sp.eye(4, format="csr"))
            operator = gb.GpuTwoLevelOperator(A)
            self.assertEqual(operator.prime.device.type, "cuda")
            self.assertEqual(operator.ns.device.type, "cuda")
            y = torch.tensor([2., -2., .1, 0.], dtype=torch.float64, device=operator.device)
            mv, rmv = operator.matvec, operator.rmatvec
            def check_cuda(fn):
                def checked(v):
                    self.assertEqual(v.device.type, "cuda")
                    result = fn(v)
                    self.assertEqual(result.device.type, "cuda")
                    return result
                return checked
            with patch.object(operator, "matvec", side_effect=check_cuda(mv)), patch.object(operator, "rmatvec", side_effect=check_cuda(rmv)), patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU transfer in CUDA solve")), patch.object(torch.Tensor, "numpy", side_effect=AssertionError("NumPy transfer in CUDA solve")):
                coef, info = gb._fista_twolevel(operator, y, .1, None, 100, 1e-10, operator.lipschitz())
            self.assertEqual(coef.device.type, "cuda")
            np.testing.assert_allclose(coef.cpu().numpy(), [1.6, -1.6, 0., 0.], atol=1e-10)
            self.assertTrue(info["converged"])

    def test_real_cuda_iterative_lstsq_residency_and_metadata(self):
        A = opt.TwoLevelSM(sp.csr_matrix(np.random.default_rng(5).normal(size=(30, 5))), sp.eye(5, format="csr"))
        truth = np.array([1., -.5, .2, 0., 0.])
        y = np.asarray(A @ truth)
        with patch.object(gb, "enabled", return_value=True):
            operator = gb.GpuTwoLevelOperator(A)
        try:
            coef, info = gb.iterative_lstsq(operator, y, atol=1e-10, btol=1e-10, maxiter=100)
            self.assertEqual(info["backend"], "gpu_resident_iterative")
            self.assertEqual(info["solver"], "GPU CGLS")
            self.assertEqual(info["device"], str(operator.device))
            self.assertTrue(info["converged"])
            np.testing.assert_allclose(coef, truth, atol=1e-8)
        finally:
            operator.close()

if __name__ == "__main__":
    unittest.main()
