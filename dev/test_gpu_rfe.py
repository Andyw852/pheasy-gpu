#!/usr/bin/env python3
"""CUDA RFE regression tests for dense, CSR and TwoLevel subset fitting."""
import os
import unittest
from unittest.mock import patch
import numpy as np
import torch
from core import gpu_backend as gb
from core import optimizer as opt

@unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware required")
class TestGpuRfe(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"PHEASY_GPU":"1", "PHEASY_GPU_LASSO":"1", "PHEASY_RFE_JACOBI":"0", "PHEASY_RFE_N_JOBS":"1", "PHEASY_MAX_CORES":"1"})
        self.env.start()
        self.previous_gpu_mode = gb.get_gpu_mode()
        gb.set_gpu_mode(True)
        rng = np.random.default_rng(19)
        self.A = rng.normal(size=(36, 8)).astype(np.float64)
        self.y = self.A @ np.array([1.0, -.5, .25, 0, 0, .1, 0, 0])
    def tearDown(self):
        gb.set_gpu_mode(self.previous_gpu_mode)
        self.env.stop()
    def test_rfe_dense_subset_solves_and_cv_prediction_use_gpu(self):
        calls = {"qr": 0, "predict": 0}
        qr, predict = gb.qr_solve, gb.predict
        def counted_qr(*args, **kwargs): calls["qr"] += 1; return qr(*args, **kwargs)
        def counted_predict(*args, **kwargs): calls["predict"] += 1; return predict(*args, **kwargs)
        model = opt.PheasyRFECV(step=.5, cv=3, min_features=2, n_jobs=1, verbose=False, patience=2)
        with patch.object(opt, "_gpu", return_value=gb), patch.object(opt, "_gpu_dense", return_value=True), patch.object(gb, "qr_solve", counted_qr), patch.object(gb, "predict", counted_predict): model.fit(self.A, self.y)
        self.assertGreater(calls["qr"], 0); self.assertGreater(calls["predict"], 0)
        self.assertEqual(model.backend_metadata_["subset_solver"], "gpu_dense")
        self.assertGreater(model.backend_metadata_["gpu_subset_solves"], 0)
        self.assertTrue(model.backend_metadata_["gpu_prediction"])
        self.assertEqual(model.backend_metadata_["orchestration"], "cpu")
        np.testing.assert_allclose(model.predict(self.A), self.A @ model.coef_, rtol=1e-10, atol=1e-10)
    def test_tsqr_dense_dominant_subset_uses_gpu_qr(self):
        calls = {"qr": 0}; qr = gb.qr_solve
        def counted_qr(*args, **kwargs): calls["qr"] += 1; return qr(*args, **kwargs)
        model = opt.PheasyRFE_OLS_TSQR(step=.5, cv=3, min_features=2, n_jobs=1, verbose=False, patience=2)
        with patch.object(opt, "_gpu", return_value=gb), patch.object(opt, "_gpu_dense", return_value=True), patch.object(gb, "qr_solve", counted_qr): model.fit(self.A, self.y)
        self.assertGreater(calls["qr"], 0)
        self.assertEqual(model.backend_metadata_["subset_solver"], "gpu_dense")
        self.assertEqual(model.backend_metadata_["orchestration"], "cpu")

    def test_opt_in_ranking_preserves_selected_support(self):
        kwargs = dict(step=.5, cv=3, min_features=2, n_jobs=1, verbose=False, patience=3, random_state=42)
        with patch.dict(os.environ, {"PHEASY_GPU_RFE_RANKING": "0"}):
            ref = opt.PheasyRFECV(**kwargs).fit(self.A, self.y)
        with patch.dict(os.environ, {"PHEASY_GPU_RFE_RANKING": "1"}):
            got = opt.PheasyRFECV(**kwargs).fit(self.A, self.y)
        np.testing.assert_array_equal(got.support_, ref.support_)
        np.testing.assert_allclose(got.coef_, ref.coef_, atol=1e-12)
        self.assertGreater(got.backend_metadata_["gpu_ranking_rounds"], 0)

    def test_qr_svd_fallback_uploads_inputs_only_once(self):
        from scipy.linalg import lstsq
        rankdef = self.A.copy()
        rankdef[:, -1] = rankdef[:, 0]
        for A, y in ((rankdef, self.y), (self.A[:4], self.y[:4]), (np.zeros_like(self.A), self.y)):
            with patch.object(gb, "_to_torch", wraps=gb._to_torch) as upload:
                coef = gb.qr_solve(A, y)
            self.assertEqual(upload.call_count, 2)
            np.testing.assert_allclose(coef, lstsq(A, y, cond=max(A.shape) * np.finfo(float).eps)[0], rtol=1e-10, atol=1e-10)

    def test_zero_importance_ties_preserve_every_subset(self):
        kwargs = dict(step=.5, cv=3, min_features=2, n_jobs=1, verbose=False, patience=3, random_state=42)
        traces = []
        for ranking in ("0", "1"):
            trace = []
            original = opt._solve_subset
            def record(A, y, rows, cols, *args, **kw):
                trace.append((tuple(cols), None if rows is None else tuple(rows)))
                return original(A, y, rows, cols, *args, **kw)
            with patch.dict(os.environ, {"PHEASY_GPU_RFE_RANKING": ranking}), patch.object(opt, "_solve_subset", record):
                model = opt.PheasyRFECV(**kwargs).fit(self.A, np.zeros(36))
            traces.append(trace)
            self.assertEqual(model.backend_metadata_["gpu_ranking_rounds"], 0)
        self.assertEqual(traces[0], traces[1])
        self.assertGreater(len({cols for cols, rows in traces[0]}), 1)

    def test_tensor_qr_keeps_output_resident_without_host_helpers(self):
        from scipy.linalg import lstsq
        rankdef = self.A.copy()
        rankdef[:, -1] = rankdef[:, 0]
        for A, y in ((self.A, self.y), (rankdef, self.y), (self.A[:4], self.y[:4])):
            At = torch.as_tensor(A, dtype=torch.float64, device="cuda")
            yt = torch.as_tensor(y, dtype=torch.float64, device="cuda")
            with patch.object(gb, "_to_numpy", side_effect=AssertionError("host download")), patch.object(gb, "_to_torch", side_effect=AssertionError("reupload")):
                coef = gb._qr_solve_tensor(At, yt)
            self.assertEqual(coef.device, At.device)
            np.testing.assert_allclose(coef.cpu().numpy(), lstsq(A, y, cond=max(A.shape)*np.finfo(float).eps)[0], atol=1e-10)

    def test_resident_subset_inputs_match_existing_path(self):
        kwargs = dict(step=.5, cv=3, min_features=2, n_jobs=1, verbose=False, patience=3, random_state=42)
        with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "0"}):
            ref = opt.PheasyRFECV(**kwargs).fit(self.A, self.y)
        with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1"}), patch.object(gb, "_qr_solve_tensor", wraps=gb._qr_solve_tensor) as solve:
            got = opt.PheasyRFECV(**kwargs).fit(self.A, self.y)
        with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1"}), patch.object(opt, "_predict_subset", side_effect=AssertionError("legacy matrix upload path")):
            checked = opt.PheasyRFECV(**kwargs).fit(self.A, self.y)
        np.testing.assert_array_equal(checked.support_, ref.support_)
        self.assertEqual(got.backend_metadata_["cv_fold_scoring"], "gpu")
        self.assertAlmostEqual(got.best_rmse_cv_, ref.best_rmse_cv_, places=12)
        self.assertTrue(got.backend_metadata_["resident_subset_inputs"])
        self.assertEqual(got.backend_metadata_["resident_row_index_uploads"], 6)
        self.assertGreater(solve.call_count, 3)
        self.assertLess(got.backend_metadata_["resident_column_index_uploads"], solve.call_count)
        self.assertEqual(got.backend_metadata_["resident_subset_builds"], got.backend_metadata_["resident_column_index_uploads"])
        np.testing.assert_array_equal(got.support_, ref.support_)
        np.testing.assert_allclose(got.coef_, ref.coef_, atol=1e-10)

    def test_resident_upload_count_lifetime_and_budget(self):
        import weakref
        import gc
        kwargs = dict(step=.5, cv=3, min_features=2, n_jobs=1, verbose=False, patience=3, random_state=42)
        for fail in (False, True):
            refs = []
            original = gb._to_torch
            def upload(*args, **kw):
                tensor = original(*args, **kw)
                refs.append(weakref.ref(tensor))
                return tensor
            # Isolate solve input transfers: legacy prediction is intentionally CPU here.
            with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1"}), patch.object(opt, "_gpu_dense", return_value=False), patch.object(gb, "_to_torch", upload):
                model = opt.PheasyRFECV(**kwargs)
                if fail:
                    with patch.object(gb, "_qr_solve_tensor", side_effect=RuntimeError("injected solve failure")):
                        with self.assertRaisesRegex(RuntimeError, "injected solve failure"):
                            model.fit(self.A, self.y)
                else:
                    model.fit(self.A, self.y)
            gc.collect()
            self.assertEqual(len(refs), 2)
            self.assertTrue(all(ref() is None for ref in refs))
        with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1"}), patch.object(opt, "_gpu_dense", return_value=False), patch.object(gb, "available_memory_bytes", return_value=1), patch.object(gb, "_to_torch", side_effect=AssertionError("unexpected upload")):
            model = opt.PheasyRFECV(**kwargs).fit(self.A, self.y)
        self.assertFalse(model.backend_metadata_["resident_subset_inputs"])
        self.assertIn("memory budget", model.backend_metadata_["resident_fallback_reason"])

    def test_cv_coefficients_never_download(self):
        original_cv = opt._cv_rmse
        checked = []
        def audit(A, y, idx, solve, splits, **kw):
            def resident_solve(cols, rows):
                with patch.object(gb, "_to_numpy", side_effect=AssertionError("CV coefficient download")):
                    coef = solve(cols, rows)
                self.assertTrue(coef.is_cuda)
                checked.append(coef.numel())
                return coef
            return original_cv(A, y, idx, resident_solve, splits, **kw)
        with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1"}), patch.object(opt, "_cv_rmse", audit):
            opt.PheasyRFECV(step=.5, cv=3, min_features=2, n_jobs=1, verbose=False, patience=3).fit(self.A, self.y)
        self.assertGreater(len(checked), 3)

    def test_resident_rankdef_and_information_criteria_parity(self):
        A = self.A.copy()
        A[:, -1] = A[:, 0]
        y = self.y + np.random.default_rng(55).normal(scale=.01, size=self.y.size)
        kwargs = dict(step=.5, cv=3, min_features=2, n_jobs=1, verbose=False, patience=3, random_state=42)
        for criterion in ("cv", "bic", "aic"):
            with self.subTest(criterion=criterion):
                with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "0", "PHEASY_TSQR_CRITERION": criterion, "PHEASY_CV_GROUP_SIZE": "6"}):
                    ref = opt.PheasyRFE_OLS_TSQR(**kwargs).fit(A, y)
                with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1", "PHEASY_TSQR_CRITERION": criterion, "PHEASY_CV_GROUP_SIZE": "6"}):
                    download = gb._to_numpy
                    def bounded_download(tensor, *args, **kw):
                        self.assertLessEqual(tensor.numel(), A.shape[1])
                        return download(tensor, *args, **kw)
                    as_tensor = torch.as_tensor
                    def no_coefficient_upload(value, *args, **kw):
                        if isinstance(value, np.ndarray) and value.dtype.kind == "f" and value.ndim == 1 and value.size <= A.shape[1]:
                            raise AssertionError("full-fit coefficients uploaded for scoring")
                        return as_tensor(value, *args, **kw)
                    with patch.object(gb, "_to_numpy", bounded_download), patch.object(torch, "as_tensor", no_coefficient_upload):
                        got = opt.PheasyRFE_OLS_TSQR(**kwargs).fit(A, y)
                self.assertTrue(got.backend_metadata_["resident_subset_inputs"])
                np.testing.assert_array_equal(got.support_, ref.support_)
                np.testing.assert_allclose(got.predict(A), ref.predict(A), atol=1e-10)
                self.assertAlmostEqual(got.best_rmse_cv_, ref.best_rmse_cv_, places=10)

    def test_partial_upload_failure_releases_matrix_and_falls_back(self):
        import weakref
        import gc
        refs = []
        original = gb._to_torch
        def upload(value, *args, **kw):
            if refs:
                raise RuntimeError("injected target upload OOM")
            tensor = original(value, *args, **kw)
            refs.append(weakref.ref(tensor))
            return tensor
        kwargs = dict(step=.5, cv=3, min_features=2, n_jobs=1, verbose=False, random_state=42)
        with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1"}), patch.object(opt, "_gpu_dense", return_value=False), patch.object(gb, "_to_torch", upload):
            model = opt.PheasyRFECV(**kwargs).fit(self.A, self.y)
        gc.collect()
        self.assertIsNone(refs[0]())
        self.assertFalse(model.backend_metadata_["resident_subset_inputs"])
        self.assertIn("injected target upload OOM", model.backend_metadata_["resident_fallback_reason"])
        np.testing.assert_allclose(model.predict(self.A), self.y, atol=1e-8)

    def test_resident_regularized_rfe_matches_gpu_reference(self):
        kwargs = dict(step=.5, cv=3, min_features=2, n_jobs=1, ridge_alpha=.2, verbose=False, random_state=42)
        with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "0"}):
            ref = opt.PheasyRFECV(**kwargs).fit(self.A, self.y)
        with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1"}), patch.object(gb, "_ridge_solve_tensor", wraps=gb._ridge_solve_tensor) as solve:
            got = opt.PheasyRFECV(**kwargs).fit(self.A, self.y)
        self.assertTrue(got.backend_metadata_["resident_subset_inputs"])
        self.assertEqual(got.backend_metadata_["resident_row_index_uploads"], 6)
        self.assertGreater(solve.call_count, 3)
        self.assertLess(got.backend_metadata_["resident_column_index_uploads"], solve.call_count)
        self.assertEqual(got.backend_metadata_["resident_subset_builds"], got.backend_metadata_["resident_column_index_uploads"])
        np.testing.assert_array_equal(got.support_, ref.support_)
        np.testing.assert_allclose(got.coef_, ref.coef_, atol=1e-10)
        self.assertAlmostEqual(got.best_rmse_cv_, ref.best_rmse_cv_, places=12)

    def test_tensor_ridge_rankdef_wide_and_transfer_boundaries(self):
        from scipy.linalg import lstsq
        rankdef = self.A.copy()
        rankdef[:, -1] = rankdef[:, 0]
        for A, y in ((rankdef, self.y), (self.A[:4], self.y[:4])):
            At = torch.as_tensor(A, dtype=torch.float64, device="cuda")
            yt = torch.as_tensor(y, dtype=torch.float64, device="cuda")
            alpha = .2
            augmented = np.vstack((A, np.sqrt(alpha) * np.eye(A.shape[1])))
            ref = lstsq(augmented, np.concatenate((y, np.zeros(A.shape[1]))))[0]
            with patch.object(gb, "_to_numpy", side_effect=AssertionError("download")), patch.object(gb, "_to_torch", side_effect=AssertionError("upload")):
                coef = gb._ridge_solve_tensor(At, yt, alpha)
            self.assertEqual(coef.device, At.device)
            np.testing.assert_allclose(coef.cpu().numpy(), ref, atol=1e-10)
            for invalid in (0., -1., float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    gb._ridge_solve_tensor(At, yt, invalid)

    def test_public_refit_clears_resident_diagnostics(self):
        env = {"PHEASY_RFE_STEP": ".5", "PHEASY_RFE_MIN_FEATURES": "2", "PHEASY_RFE_N_JOBS": "1", "PHEASY_RFE_RIDGE_ALPHA": ".2", "PHEASY_GPU_RFE_RESIDENT": "1"}
        model = opt.Optimizer("RFE", cv=3, rand_seed=42, use_gpu=True)
        with patch.dict(os.environ, env):
            model.fit(self.A, self.y)
        metadata = model.results["backend_metadata"]
        self.assertTrue(metadata["resident_subset_inputs"])
        self.assertEqual(metadata["cv_fold_scoring"], "gpu")
        pred = model.predict(self.A)
        env["PHEASY_GPU_RFE_RESIDENT"] = "0"
        with patch.dict(os.environ, env):
            model.fit(self.A, self.y)
        self.assertFalse(model.results["backend_metadata"]["resident_subset_inputs"])
        self.assertEqual(model.results["backend_metadata"]["cv_fold_scoring"], "cpu")
        self.assertIsNone(model.results["backend_metadata"]["resident_fallback_reason"])
        np.testing.assert_allclose(model.predict(self.A), pred, atol=1e-10)

    def test_index_cache_is_included_in_resident_memory_gate(self):
        m, n = self.A.shape
        old_budget = 8 * (9*self.A.size + 16*n*n + 4*m) + 64*1024**2
        with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1", "PHEASY_GPU_MEM_FRACTION": "1"}), patch.object(gb, "available_memory_bytes", return_value=old_budget), patch.object(opt, "_gpu_dense", return_value=False), patch.object(gb, "_to_torch", side_effect=AssertionError("upload exceeds index-inclusive budget")):
            got = opt.PheasyRFECV(step=.5, cv=3, min_features=2, n_jobs=1, verbose=False).fit(self.A, self.y)
        self.assertFalse(got.backend_metadata_["resident_subset_inputs"])
        self.assertEqual(got.backend_metadata_["resident_index_cache_budget_bytes"], 8*(3*m+n))
        exact_budget = old_budget + 8*(3*m+n)
        with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1", "PHEASY_GPU_MEM_FRACTION": "1"}), patch.object(gb, "available_memory_bytes", return_value=exact_budget), patch.object(opt, "_gpu_dense", return_value=False):
            accepted = opt.PheasyRFECV(step=.5, cv=3, min_features=2, n_jobs=1, verbose=False).fit(self.A, self.y)
        self.assertTrue(accepted.backend_metadata_["resident_subset_inputs"])
        np.testing.assert_allclose(accepted.predict(self.A), got.predict(self.A), atol=1e-8)

    def test_dense_jacobi_option_does_not_disable_residency(self):
        kwargs = dict(step=.5, cv=3, min_features=2, n_jobs=1, ridge_alpha=.2, verbose=False, random_state=42)
        models = []
        for resident in ("0", "1"):
            with patch.dict(os.environ, {"PHEASY_RFE_JACOBI": "1", "PHEASY_GPU_RFE_RESIDENT": resident}):
                models.append(opt.PheasyRFECV(**kwargs).fit(self.A, self.y))
        ref, got = models
        self.assertTrue(got.backend_metadata_["resident_subset_inputs"])
        self.assertFalse(got.backend_metadata_["jacobi_applied"])
        np.testing.assert_array_equal(got.support_, ref.support_)
        np.testing.assert_allclose(got.coef_, ref.coef_, atol=1e-10)

    def test_failed_support_allocation_allows_clean_refit(self):
        model = opt.PheasyRFECV(step=.5, cv=3, min_features=2, n_jobs=1, verbose=False, random_state=42)
        original = torch.Tensor.index_select
        calls = []
        def fail_next_support(tensor, dim, index):
            if dim == 1:
                calls.append(index.numel())
                if len(calls) == 2:
                    raise RuntimeError("injected support allocation failure")
            return original(tensor, dim, index)
        with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1"}):
            with patch.object(torch.Tensor, "index_select", fail_next_support):
                with self.assertRaisesRegex(RuntimeError, "support allocation failure"):
                    model.fit(self.A, self.y)
            model.fit(self.A, self.y)
        self.assertTrue(model.backend_metadata_["resident_subset_inputs"])
        np.testing.assert_allclose(model.predict(self.A), self.y, atol=1e-8)

    def test_resident_importance_matches_host_with_and_without_ties(self):
        kwargs = dict(step=.5, cv=3, min_features=2, n_jobs=1, verbose=False, random_state=42)
        for y in (self.y, np.zeros_like(self.y)):
            models = []
            for ranking in ("0", "1"):
                with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1", "PHEASY_GPU_RFE_RANKING": ranking}):
                    models.append(opt.PheasyRFECV(**kwargs).fit(self.A, y))
            ref, got = models
            self.assertGreater(got.backend_metadata_["gpu_importance_rounds"], 0)
            np.testing.assert_array_equal(got.support_, ref.support_)
            np.testing.assert_allclose(got.coef_, ref.coef_, atol=1e-10)

    def test_resident_ranking_downloads_coefficients_only_for_final_output(self):
        y = self.y + np.random.default_rng(382).normal(scale=.01, size=self.y.size)
        import scipy.sparse as sp
        inputs = (self.A, sp.csr_matrix(self.A), opt.TwoLevelSM(sp.csr_matrix(self.A), sp.eye(8, format="csr")))
        for A in inputs:
            for verbose in (False, True):
                with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1", "PHEASY_GPU_RFE_RANKING": "1"}), patch.object(gb, "_to_numpy", wraps=gb._to_numpy) as download:
                    model = opt.PheasyRFECV(step=.5, cv=3, min_features=2, n_jobs=1, verbose=verbose, random_state=42).fit(A, y)
                self.assertGreater(model.backend_metadata_["gpu_ranking_rounds"], 0)
                self.assertEqual(model.backend_metadata_["gpu_ranking_rounds"], model.backend_metadata_["gpu_importance_rounds"])
                self.assertEqual(download.call_count, 1)

    def test_public_csr_resident_parity(self):
        import scipy.sparse as sp
        env = {"PHEASY_RFE_STEP": ".5", "PHEASY_RFE_MIN_FEATURES": "2",
               "PHEASY_RFE_RIDGE_ALPHA": ".2", "PHEASY_LSQR_ATOL": "1e-11",
               "PHEASY_LSQR_BTOL": "1e-11", "PHEASY_GPU_RFE_RESIDENT": "1"}
        with patch.dict(os.environ, env):
            ref = opt.Optimizer("RFE", cv=3, rand_seed=42, use_gpu=False)
            ref.fit(self.A, self.y)
            got = opt.Optimizer("RFE", cv=3, rand_seed=42, use_gpu=True)
            with patch.object(sp.csr_matrix, "toarray", side_effect=AssertionError("densification")):
                got.fit(sp.csr_matrix(self.A), self.y)
        self.assertFalse(got.results["backend_metadata"]["jacobi_applied"])
        self.assertTrue(got.results["backend_metadata"]["resident_subset_inputs"])
        self.assertEqual(got.results["backend_metadata"]["subset_solver"], "gpu_resident_iterative")
        np.testing.assert_allclose(got.results["coef"], ref.results["coef"], atol=1e-8)

    def test_public_twolevel_grouped_cv_and_ic_without_densification(self):
        import scipy.sparse as sp
        rng = np.random.default_rng(77)
        prime = sp.csr_matrix(rng.normal(size=(48, 12)))
        ns = sp.csr_matrix(rng.normal(size=(12, 8)) * np.geomspace(.1, 10, 8))
        dense = prime.toarray() @ ns.toarray()
        y = dense @ np.array([1., -.5, .25, 0, 0, .1, 0, 0]) + rng.normal(scale=.01, size=48)
        env = {"PHEASY_GPU_RFE_RESIDENT": "1", "PHEASY_CV_GROUP_SIZE": "4",
               "PHEASY_TSQR_MIN_FEATURES": "2",
               "PHEASY_RFE_STEP": ".5", "PHEASY_RFE_MIN_FEATURES": "2",
               "PHEASY_RFE_RIDGE_ALPHA": ".2", "PHEASY_RFE_JACOBI": "1",
               "PHEASY_LSQR_ATOL": "1e-11", "PHEASY_LSQR_BTOL": "1e-11"}
        for method, criterion in (("RFE", "cv"), ("RFE-OLS-TSQR", "bic"), ("RFE-OLS-TSQR", "aic")):
            with self.subTest(method=method, criterion=criterion), patch.dict(os.environ, dict(env, PHEASY_TSQR_CRITERION=criterion)):
                ref = opt.Optimizer(method, cv=3, rand_seed=42, use_gpu=False)
                ref.fit(dense, y)
                got = opt.Optimizer(method, cv=3, rand_seed=42, use_gpu=True)
                operator = opt.TwoLevelSM(prime, ns)
                with patch.object(sp.csr_matrix, "toarray", side_effect=AssertionError("densification")), patch.object(opt, "_predict_subset", side_effect=AssertionError("CPU subset prediction")):
                    got.fit(operator, y)
                meta = got.results["backend_metadata"]
                self.assertTrue(meta["jacobi_applied"])
                self.assertEqual(meta["resident_input_kind"], "twolevel")
                self.assertEqual(meta["cv_fold_scoring"], "gpu")
                self.assertEqual(meta["resident_row_index_uploads"], 6)
                self.assertEqual(len(meta["iterative_diagnostics"]), meta["gpu_subset_solves"])
                for info in meta["iterative_diagnostics"]:
                    self.assertTrue(info["converged"])
                    self.assertEqual(info["solver"], "GPU CGLS")
                    self.assertEqual(info["atol"], 1e-11)
                    self.assertGreater(info["itn"], 0)
                    if info["fit_scope"] == "fold":
                        self.assertEqual(info["n_samples"], 32)
                np.testing.assert_array_equal(got.results["coef"] != 0, ref.results["coef"] != 0)
                np.testing.assert_allclose(got.results["coef"], ref.results["coef"], rtol=1e-7, atol=1e-8)
                self.assertAlmostEqual(got.metrics["rmse_path_mean"], ref.metrics["rmse_path_mean"], places=7)

    def test_public_sparse_nonconvergence_aborts_fit(self):
        import scipy.sparse as sp
        for A in (sp.csr_matrix(self.A), opt.TwoLevelSM(sp.csr_matrix(self.A), sp.eye(8, format="csr"))):
            with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1", "PHEASY_LSQR_MAXITER": "1", "PHEASY_LSQR_ATOL": "0", "PHEASY_LSQR_BTOL": "0", "PHEASY_RFE_MIN_FEATURES": "2"}):
                model = opt.Optimizer("RFE", cv=3, use_gpu=True)
                with self.assertWarns(RuntimeWarning), self.assertRaisesRegex(RuntimeError, "Resident subset solve did not converge"):
                    model.fit(A, self.y)
                self.assertNotIn("coef", model.results)

    def test_public_sparse_zero_target_and_tied_importance(self):
        import scipy.sparse as sp
        env = {"PHEASY_GPU_RFE_RESIDENT": "1", "PHEASY_GPU_RFE_RANKING": "1",
               "PHEASY_RFE_STEP": ".5", "PHEASY_RFE_MIN_FEATURES": "2",
               "PHEASY_LSQR_ATOL": "0", "PHEASY_LSQR_BTOL": "0",
               "PHEASY_RFE_JACOBI": "1", "PHEASY_RFE_RIDGE_ALPHA": "0"}
        zero = np.zeros_like(self.y)
        with patch.dict(os.environ, env):
            ref = opt.Optimizer("RFE", cv=3, rand_seed=42, use_gpu=False)
            ref.fit(self.A, zero)
            for A in (sp.csr_matrix(self.A), opt.TwoLevelSM(sp.csr_matrix(self.A), sp.eye(8, format="csr"))):
                got = opt.Optimizer("RFE", cv=3, rand_seed=42, use_gpu=True)
                transfers = []
                as_tensor = torch.as_tensor
                def audit_transfer(value, *args, **kw):
                    if (isinstance(value, np.ndarray) and value.dtype == np.float64 and value.shape == (8,)
                            and np.allclose(value, np.linalg.norm(self.A, axis=0))):
                        transfers.append(value.copy())
                    return as_tensor(value, *args, **kw)
                with patch.object(torch, "as_tensor", audit_transfer):
                    got.fit(A, zero)
                self.assertEqual(len(transfers), 1)
                np.testing.assert_array_equal(got.results["coef"], np.zeros(8))
                np.testing.assert_array_equal(got._model.support_, ref._model.support_)
                self.assertEqual(got.metrics["rmse_path_mean"], 0.)
                meta = got.results["backend_metadata"]
                self.assertTrue(meta["resident_subset_inputs"])
                self.assertGreater(meta["gpu_importance_rounds"], 0)
                self.assertEqual(meta["gpu_ranking_rounds"], 0)
                self.assertEqual({info["n_features"] for info in meta["iterative_diagnostics"]}, {8, 4, 2})
                for info in meta["iterative_diagnostics"]:
                    self.assertTrue(info["converged"])
                    self.assertEqual(info["itn"], 0)
                    self.assertEqual(info["normar"], 0.)

    def test_public_sparse_setup_failure_reports_fallback(self):
        import scipy.sparse as sp
        with patch.dict(os.environ, {"PHEASY_GPU_RFE_RESIDENT": "1", "PHEASY_RFE_MIN_FEATURES": "2"}), patch.object(gb, "GpuCSRResidentOperator", side_effect=MemoryError("injected budget failure")):
            model = opt.Optimizer("RFE", cv=3, use_gpu=True)
            model.fit(sp.csr_matrix(self.A), self.y)
        meta = model.results["backend_metadata"]
        self.assertFalse(meta["resident_subset_inputs"])
        self.assertIn("injected budget failure", meta["resident_fallback_reason"])
        self.assertEqual(meta["iterative_diagnostics"], [])

if __name__ == "__main__": unittest.main()
