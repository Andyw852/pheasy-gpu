#!/usr/bin/env python3
import os
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
from core import optimizer as opt

class TestOperatorRidgeGPU(unittest.TestCase):
    def test_subset_norm_cache_is_local_to_view(self):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        with patch.dict(os.environ, {"PHEASY_GPU": "1"}):
            base = gb.GpuCSRResidentOperator(sp.diags([1., 10., 100.], format="csr"))
            try:
                base._norma = 999.
                a = gb.GpuSubsetOperator(base, [0])
                b = gb.GpuSubsetOperator(base, [1])
                with patch.object(gb, "_operator_norm_estimate", wraps=gb._operator_norm_estimate) as estimate:
                    self.assertAlmostEqual(a.norm_estimate(), 1.)
                    self.assertAlmostEqual(a.norm_estimate(), 1.)
                    self.assertAlmostEqual(b.norm_estimate(), 10.)
                    self.assertEqual(estimate.call_count, 2)
            finally:
                base.close()

    def test_augmented_norm_cache_is_local_to_alpha(self):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        with patch.dict(os.environ, {"PHEASY_GPU": "1"}):
            base = gb.GpuCSRResidentOperator(sp.eye(3, format="csr") * 2.)
            seen = []
            solve = gb._iterative_lstsq_tensor
            def inspect(A, y, *args):
                first = A.norm_estimate()
                with patch.object(gb, "_operator_norm_estimate", side_effect=AssertionError("cache missed")):
                    self.assertEqual(A.norm_estimate(), first)
                seen.append(first)
                return solve(A, y, *args)
            try:
                with patch.object(gb, "_iterative_lstsq_tensor", side_effect=inspect):
                    for alpha in (1., 12.):
                        gb._iterative_ridge_tensor(base, np.ones(3), alpha)
                np.testing.assert_allclose(seen, np.sqrt([5., 16.]), rtol=1e-12)
            finally:
                base.close()

    def test_cgls_stopping_is_invariant_to_joint_scaling(self):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        class Operator:
            def matvec(self, x): return self.matrix @ x
            def rmatvec(self, y): return self.matrix.T @ y
        for scale in (1., 1e-12, 1e12):
            A = Operator()
            A.torch, A.device = torch, torch.device("cuda:0")
            A.matrix = torch.eye(3, dtype=torch.float64, device=A.device) * scale
            A.shape = (3, 3)
            target = torch.tensor([1., 2., -3.], dtype=torch.float64, device=A.device)
            coef, info = gb._iterative_lstsq_tensor(A, target * scale)
            self.assertTrue(info["converged"])
            torch.testing.assert_close(coef, target, rtol=1e-10, atol=1e-10)
            self.assertAlmostEqual(info["norma"] / scale, 1., places=12)
            self.assertEqual(info["norma_estimator"], "spectral_power_10")
            self.assertTrue(info["normar"] <= 1e-8 * info["norma"] * info["normr"]
                            or info["normr"] <= 1e-8 * (info["normb"] + info["norma"] * info["normx"]))

    def test_cgls_inconsistent_system_matches_cpu_lsmr(self):
        import torch
        from scipy.sparse.linalg import lsmr
        from core import gpu_backend as gb
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        rng = np.random.default_rng(72)
        matrix = rng.normal(size=(80, 12))
        y = matrix @ rng.normal(size=12) + rng.normal(scale=.01, size=80)
        class Operator:
            def matvec(self, x): return self.matrix @ x
            def rmatvec(self, y): return self.matrix.T @ y
        for scale in (1e-9, 1., 1e9):
            A = Operator()
            A.torch, A.device = torch, torch.device("cuda:0")
            A.matrix = torch.as_tensor(matrix * scale, device=A.device)
            A.shape = matrix.shape
            cpu = lsmr(matrix * scale, y * scale, atol=1e-8, btol=1e-8, conlim=0, maxiter=100)
            gpu, info = gb.iterative_lstsq(A, y * scale)
            self.assertTrue(info["converged"])
            self.assertIn(cpu[1], (1, 2))
            np.testing.assert_allclose(gpu, cpu[0], rtol=1e-7, atol=1e-8)
            residual = (y - matrix @ gpu) * scale
            self.assertAlmostEqual(info["normr"] / np.linalg.norm(residual), 1., places=9)
            self.assertEqual(info["residual_certificate"], "recomputed_y_minus_Ax")

    def test_full_column_norms_do_not_change_operator(self):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        prime = sp.csr_matrix([[1., 2.], [3., -1.], [0., 4.]])
        ns = sp.csr_matrix([[2., 0., 0.], [1., .01, 0.]])
        expected = np.linalg.norm((prime @ ns).toarray(), axis=0)
        with patch.dict(os.environ, {"PHEASY_GPU": "1"}):
            A = gb.GpuTwoLevelOperator(opt.TwoLevelSM(prime, ns))
            try:
                x = torch.ones(3, dtype=torch.float64, device=A.device)
                before = A.matvec(x).clone()
                norms = A.col_norms()
                self.assertEqual(norms.device.type, "cuda")
                np.testing.assert_allclose(norms.cpu().numpy(), expected)
                torch.testing.assert_close(A.matvec(x), before, rtol=0, atol=0)
                A.normalize()
                np.testing.assert_allclose(A.col_norms().cpu().numpy(), [1., 1., 0.])
                normalized = A.matvec(x).clone()
                A.normalize()
                torch.testing.assert_close(A.matvec(x), normalized)
            finally:
                A.close()

    def test_extra_workspace_rejected_before_any_upload(self):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        matrix = sp.eye(5, format="csr")
        twolevel = opt.TwoLevelSM(matrix, matrix)
        for adapter, source in ((gb.GpuCSRResidentOperator, matrix), (gb.GpuTwoLevelOperator, twolevel)):
            with patch.dict(os.environ, {"PHEASY_GPU": "1", "PHEASY_GPU_MEM_FRACTION": "1"}):
                probe = adapter(source)
                budget = probe.estimated_peak_bytes
                probe.close()
                with patch.object(gb, "_device_free_bytes", return_value=budget), patch.object(torch, "as_tensor", side_effect=AssertionError("upload before budget rejection")):
                    with self.assertRaises(MemoryError):
                        adapter(source, extra_workspace_bytes=1)
                with patch.object(gb, "_device_free_bytes", return_value=budget+128):
                    accepted = adapter(source, extra_workspace_bytes=128)
                self.assertEqual(accepted.estimated_peak_bytes, budget+128)
                accepted.close()

    def test_qr_resource_failure_does_not_retry_svd(self):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        A = torch.eye(3, dtype=torch.float64, device="cuda")
        y = torch.ones(3, dtype=torch.float64, device="cuda")
        for failure in (torch.cuda.OutOfMemoryError("injected OOM"), MemoryError("injected host allocation failure")):
            with patch.object(torch.linalg, "qr", side_effect=failure), patch.object(torch.linalg, "svd", side_effect=AssertionError("SVD retry after resource failure")):
                with self.assertRaises(type(failure)):
                    gb._qr_solve_tensor(A, y)
            with patch.object(torch.linalg, "cholesky", side_effect=failure), patch.object(torch.linalg, "solve", side_effect=AssertionError("solve retry after resource failure")):
                with self.assertRaises(type(failure)):
                    gb._ridge_solve_tensor(A, y, .2)
        with patch.object(torch.linalg, "cholesky", side_effect=RuntimeError("injected numerical failure")):
            coef = gb._ridge_solve_tensor(A, y, .2)
        torch.testing.assert_close(coef, y / 1.2)

    def test_resident_ridge_matches_cpu_reference(self):
        try:
            import torch
            if not torch.cuda.is_available(): self.skipTest("CUDA required")
        except ImportError:
            self.skipTest("Torch required")
        rng=np.random.default_rng(31)
        base=sp.csr_matrix(rng.normal(size=(36,7)))
        A=opt.TwoLevelSM(base,sp.eye(7,format="csr")); truth=rng.normal(size=7)
        y=np.asarray(A@truth)
        with patch.dict(os.environ,{"PHEASY_GPU":"1","PHEASY_GPU_RIDGE_RESIDENT":"1"}):
            gpu=opt._ridge_solve(A,y,.2)
        cpu=opt._ridge_solve(A,y,.2)
        np.testing.assert_allclose(gpu,cpu,rtol=2e-6,atol=2e-8)
        model=opt.Optimizer("RIDGE",alpha=[.01,.2],cv=2,use_gpu=True)
        with patch.dict(os.environ,{"PHEASY_GPU":"1","PHEASY_GPU_RIDGE_RESIDENT":"1"}):
            model.fit(A,y)
        self.assertIn(model.results["execution_backend"],["gpu_twolevel_ridge_resident","gpu_resident_iterative"])
        self.assertEqual(model.results["regularized_solver_info"]["device"],"cuda:0")
    def test_flag_off_preserves_cpu_operator_path(self):
        A=opt.TwoLevelSM(sp.eye(6,format="csr"),sp.eye(6,format="csr")); y=np.arange(6.)
        with patch.dict(os.environ,{"PHEASY_GPU_RIDGE_RESIDENT":"0"}):
            with patch.object(opt,"_lsmr",wraps=opt._lsmr) as lsmr:
                opt._ridge_solve(A,y,.1)
                self.assertTrue(lsmr.called)
    def test_true_residual_can_accept_after_iteration_limit(self):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        # Fault-injection operator isolates the two certificate branches;
        # this is a diagnostic test, not a physical least-squares fixture.
        class Operator:
            shape = (2, 2)
            device = torch.device("cuda:0")
            calls = 0
            def norm_estimate(self): return 1.
            def matvec(self, x):
                self.calls += 1
                return x * x.new_tensor([1., 2.]) if self.calls == 1 else x.new_tensor([1., 2.])
            def rmatvec(self, x): return x
        A = Operator()
        A.torch = torch
        coef, info = gb._iterative_lstsq_tensor(A, [1., 2.], maxiter=1, btol=0., atol=0.)
        self.assertTrue(info["converged"])
        self.assertEqual(info["stop_reason"], "converged_on_true_residual")
        self.assertEqual(info["normr"], 0.)
        self.assertEqual(info["n_iter"], 1)

    def test_tensor_cgls_preserves_device_and_diagnostics(self):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        class Operator:
            shape = (6, 6)
            device = torch.device("cuda:0")
            def matvec(self, x): return x
            def rmatvec(self, x): return x
        A = Operator()
        A.torch = torch
        y = torch.arange(6, dtype=torch.float64, device=A.device)
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("coefficient download")):
            coef, info = gb._iterative_lstsq_tensor(A, y)
        self.assertEqual(coef.device, y.device)
        torch.testing.assert_close(coef, y)
        host, host_info = gb.iterative_lstsq(A, y)
        np.testing.assert_allclose(host, y.cpu().numpy())
        for invalid in (float("inf"), float("-inf"), float("nan")):
            bad = y.clone()
            bad[0] = invalid
            with self.assertRaisesRegex(ValueError, "target must be finite"):
                gb._iterative_lstsq_tensor(A, bad)
        with self.assertWarnsRegex(RuntimeWarning, "reason=invalid_search_direction"):
            _, overflow_info = gb._iterative_lstsq_tensor(A, torch.full_like(y, 1e308))
        self.assertFalse(overflow_info["converged"])
        self.assertEqual(overflow_info["stop_reason"], "invalid_search_direction")
        self.assertEqual(info["stop_reason"], "converged")
        self.assertEqual(info, host_info)
        self.assertTrue(info["converged"])
        zeros, zero_info = gb._iterative_lstsq_tensor(A, torch.zeros_like(y), atol=0., btol=0.)
        self.assertTrue(zero_info["converged"])
        self.assertEqual(zero_info["n_iter"], 0)
        self.assertEqual(zero_info["normar"], 0.)
        torch.testing.assert_close(zeros, torch.zeros_like(y))
        for options in ({"atol": float("inf")}, {"btol": float("nan")}, {"maxiter": 1.5}, {"maxiter": float("inf")}):
            with self.assertRaises(ValueError):
                gb._iterative_lstsq_tensor(A, y, **options)

    def test_resident_subset_operator_adjoint_and_solve(self):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        rng = np.random.default_rng(123)
        matrix = rng.normal(size=(20, 7))
        class Operator:
            shape = matrix.shape
            device = torch.device("cuda:0")
            def matvec(self, x): return self.matrix @ x
            def rmatvec(self, y): return self.matrix.T @ y
        base = Operator()
        base.torch = torch
        base.matrix = torch.as_tensor(matrix, device=base.device)
        for bad in ([-1], [7], [[1, 2]], [1.5], [True], [1+0j]):
            with self.assertRaises(ValueError):
                gb.GpuSubsetOperator(base, bad)
        columns = torch.tensor([5, 1, 3], device=base.device)
        owned = gb.GpuSubsetOperator(base, columns)
        columns.fill_(0)
        self.assertEqual(owned.columns.tolist(), [5, 1, 3])
        for rows in (None, [9, 2, 5, 1, 7, 12], [9, 2, 2, 5]):
            cols = [5, 1, 3]
            view = gb.GpuSubsetOperator(base, cols, rows)
            ref = matrix[:, cols] if rows is None else matrix[rows][:, cols]
            x = torch.ones(3, dtype=torch.float64, device=base.device)
            y = torch.ones(ref.shape[0], dtype=torch.float64, device=base.device)
            with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")):
                forward, adjoint = view.matvec(x), view.rmatvec(y)
                coef, info = gb._iterative_lstsq_tensor(view, forward, atol=1e-11)
            np.testing.assert_allclose(forward.cpu().numpy(), ref @ np.ones(3))
            np.testing.assert_allclose(adjoint.cpu().numpy(), ref.T @ np.ones(ref.shape[0]))
            self.assertTrue(info["converged"])
            torch.testing.assert_close(coef, x)

    def test_twolevel_subset_ridge_without_densification(self):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        rng = np.random.default_rng(61)
        prime = sp.csr_matrix(rng.normal(size=(30, 9)))
        ns = sp.csr_matrix(rng.normal(size=(9, 7)))
        A = opt.TwoLevelSM(prime, ns)
        rows, cols = [1, 4, 8, 12, 17, 21], [5, 0, 2]
        dense = (prime @ ns).toarray()[rows][:, cols]
        y = rng.normal(size=len(rows))
        ref = np.linalg.solve(dense.T @ dense + .2*np.eye(3), dense.T @ y)
        with patch.dict(os.environ, {"PHEASY_GPU": "1"}), patch.object(sp.csr_matrix, "toarray", side_effect=AssertionError("densification")):
            base = gb.GpuTwoLevelOperator(A)
            view = gb.GpuSubsetOperator(base, cols, rows)
            target = torch.as_tensor(y, device=base.device)
            with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("download")):
                coef, info = gb._iterative_ridge_tensor(view, target, .2, atol=1e-11, btol=1e-11)
        self.assertTrue(info["converged"])
        np.testing.assert_allclose(coef.cpu().numpy(), ref, atol=1e-9)

    def test_csr_adapter_subset_and_normalization(self):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        matrix = np.random.default_rng(72).normal(size=(18, 5))
        with patch.dict(os.environ, {"PHEASY_GPU": "1"}), patch.object(sp.csr_matrix, "toarray", side_effect=AssertionError("densification")):
            base = gb.GpuCSRResidentOperator(sp.csr_matrix(matrix))
            base.normalize()
            view = gb.GpuSubsetOperator(base, [3, 0], [1, 4, 7, 10])
            x = torch.ones(2, dtype=torch.float64, device=base.device)
            result = view.matvec(x)
            adjoint = view.rmatvec(torch.ones(4, dtype=torch.float64, device=base.device))
        ref = (matrix / np.linalg.norm(matrix, axis=0))[[1, 4, 7, 10]][:, [3, 0]]
        np.testing.assert_allclose(result.cpu().numpy(), ref @ np.ones(2))
        np.testing.assert_allclose(adjoint.cpu().numpy(), ref.T @ np.ones(4))
        base.close()
        self.assertIsNone(base.prime)

    def test_subset_column_preconditioner_preserves_adjoint(self):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        matrix = np.random.default_rng(91).normal(size=(12, 4))
        with patch.dict(os.environ, {"PHEASY_GPU": "1"}):
            base = gb.GpuCSRResidentOperator(sp.csr_matrix(matrix))
        view = gb.GpuSubsetOperator(base, [3, 1], column_scale=[2., 0.])
        x = torch.ones(2, dtype=torch.float64, device=base.device)
        y = torch.arange(12, dtype=torch.float64, device=base.device)
        ref = matrix[:, [3, 1]] / [2., 1.]
        np.testing.assert_allclose(view.matvec(x).cpu().numpy(), ref @ np.ones(2))
        np.testing.assert_allclose(view.rmatvec(y).cpu().numpy(), ref.T @ np.arange(12))
        coef, info = gb._iterative_ridge_tensor(view, y, .3, atol=1e-11, penalty_scale=view.column_scale)
        physical = coef / view.column_scale
        raw = matrix[:, [3, 1]]
        expected = np.linalg.solve(raw.T @ raw + .3*np.eye(2), raw.T @ np.arange(12))
        self.assertTrue(info["converged"])
        np.testing.assert_allclose(physical.cpu().numpy(), expected, atol=1e-9)
        for invalid in ([1.], [-1., 1.], [float("nan"), 1.]):
            with self.assertRaises(ValueError):
                gb.GpuSubsetOperator(base, [3, 1], column_scale=invalid)

    def test_subset_solve_physical_coefficients_and_nonconvergence(self):
        import torch
        from core import gpu_backend as gb
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        matrix = np.random.default_rng(14).normal(size=(20, 5))
        target = np.arange(20.)
        rows, cols = [2, 4, 6, 8, 10], [4, 1, 0]
        raw = matrix[rows][:, cols]
        with patch.dict(os.environ, {"PHEASY_GPU": "1"}):
            base = gb.GpuCSRResidentOperator(sp.csr_matrix(matrix))
        for alpha in (0., .2):
            coef, info = gb.solve_resident_subset(base, target, cols, rows, [2., 3., 4.], ridge_alpha=alpha, atol=1e-11)
            ref = np.linalg.solve(raw.T @ raw + alpha*np.eye(3), raw.T @ target[rows])
            np.testing.assert_allclose(coef.cpu().numpy(), ref, atol=1e-9)
            self.assertTrue(info["converged"])
        with self.assertWarnsRegex(RuntimeWarning, "reason=iteration_limit"):
            with self.assertRaisesRegex(RuntimeError, "did not converge") as failed:
                gb.solve_resident_subset(base, target, cols, rows, maxiter=1, atol=0., btol=0.)
        self.assertIn("'fit_scope': 'fold'", str(failed.exception))
        self.assertIn("'n_samples': 5", str(failed.exception))
        self.assertIn("'n_features': 3", str(failed.exception))

class TestOperatorOLSFallback(unittest.TestCase):
    def test_failed_upload_reports_reason_and_next_fit_clears_it(self):
        from core import gpu_backend as gb
        A = opt.TwoLevelSM(sp.eye(6, format="csr"), sp.eye(6, format="csr"))
        y = np.arange(6.)
        model = opt.Optimizer("OLS", use_gpu=True)
        # PHEASY_OLS_JACOBI=0 is required for this scenario to exist at all: the
        # matrix-free default is Jacobi ON, and the resident OLS branch skips
        # itself (setting fallback_reason = "...ridge/Jacobi options") whenever
        # Jacobi or a ridge is requested, so the injected upload failure below
        # would never be attempted.  Same env as
        # test_resident_ols_honors_limits_and_ridge_option, which already knew
        # this.  The assertion intent is unchanged: a *failed* resident upload
        # must be reported in fallback_reason.
        with patch.dict(os.environ, {"PHEASY_GPU_OLS_RESIDENT": "1", "PHEASY_OLS_JACOBI": "0"}), patch.object(gb, "enabled", return_value=True), patch.object(gb, "GpuTwoLevelOperator", side_effect=RuntimeError("injected upload failure")):
            model.fit(A, y)
        self.assertEqual(model.results["execution_backend"], "cpu_lsmr")
        self.assertIn("injected upload failure", model.results["fallback_reason"])
        np.testing.assert_allclose(model.predict(A), y, atol=1e-8)
        with patch.dict(os.environ, {"PHEASY_GPU_OLS_RESIDENT": "0"}):
            model.fit(A, y)
        self.assertNotIn("fallback_reason", model.results)
        self.assertNotIn("execution_backend", model.results)

    def test_matrix_free_default_skips_resident_ols(self):
        """Default (Jacobi unset) must NOT reach the resident GPU OLS.

        Jacobi defaults ON for matrix-free input, and the resident OLS branch
        implements neither Jacobi nor a ridge, so it skips itself and sets
        fallback_reason to the ridge/Jacobi message before it ever builds a
        GpuTwoLevelOperator.  So the resident OLS is unreachable under the
        default configuration and every OLS fit lands on CPU LSMR -- a
        deliberate trade-off (Jacobi is what keeps LSMR from stalling at
        istop=7 on 1e5-unknown systems), but a silent one.  Pin it down: if a
        later change makes the resident OLS reachable by default, this test
        fails and the GPU.md / README claims have to be re-measured.
        """
        from core import gpu_backend as gb
        A = opt.TwoLevelSM(sp.eye(6, format="csr"), sp.eye(6, format="csr"))
        y = np.arange(6.)
        model = opt.Optimizer("OLS", use_gpu=True)
        with patch.dict(os.environ, {"PHEASY_GPU_OLS_RESIDENT": "1"}):
            os.environ.pop("PHEASY_OLS_JACOBI", None)   # the default spelling
            with patch.object(gb, "enabled", return_value=True), \
                    patch.object(gb, "GpuTwoLevelOperator") as operator:
                model.fit(A, y)
                operator.assert_not_called()
        self.assertEqual(model.results["execution_backend"], "cpu_lsmr")
        self.assertIn("ridge/Jacobi", model.results["fallback_reason"])

    def test_resident_ols_honors_limits_and_ridge_option(self):
        from core import gpu_backend as gb
        A = opt.TwoLevelSM(sp.eye(6, format="csr"), sp.eye(6, format="csr"))
        y = np.arange(6.)
        model = opt.Optimizer("OLS", use_gpu=True)
        env = {"PHEASY_GPU_OLS_RESIDENT": "1", "PHEASY_GPU_TSQR": "0", "PHEASY_OLS_ATOL": "2e-9", "PHEASY_OLS_BTOL": "3e-9", "PHEASY_OLS_MAXITER": "17", "PHEASY_OLS_RIDGE": "0", "PHEASY_OLS_JACOBI": "0"}
        with patch.dict(os.environ, env), patch.object(gb, "enabled", return_value=True), patch.object(gb, "GpuTwoLevelOperator") as operator, patch.object(gb, "iterative_lstsq", return_value=(y, {"itn": 1, "backend": "gpu_test"})) as solve:
            model.fit(A, y)
            self.assertEqual(solve.call_args.kwargs, dict(atol=2e-9, btol=3e-9, maxiter=17))
            operator.return_value.close.assert_called_once()
        env["PHEASY_OLS_RIDGE"] = "0.2"
        with patch.dict(os.environ, env), patch.object(gb, "GpuTwoLevelOperator") as operator:
            model.fit(A, y)
            operator.assert_not_called()
        np.testing.assert_allclose(model.predict(A), y / (1 + .2 * 6), atol=1e-8)
        self.assertIn("ridge/Jacobi", model.results["fallback_reason"])

if __name__ == "__main__": unittest.main()
