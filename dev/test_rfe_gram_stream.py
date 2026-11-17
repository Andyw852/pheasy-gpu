#!/usr/bin/env python3
"""[RFE-GRAM-STREAM] regressions for the streaming per-fold Gram engine.

CPU only.  The GPU accumulator tests run the real torch code on torch's CPU
device (PHEASY_RFE_GRAM_GPU_DEVICES=cpu) and are skipped without torch.

Run from the repository root:  python3 dev/test_rfe_gram_stream.py
"""
import contextlib
import importlib.util
import io
import itertools
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import scipy.sparse as sp

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from core import optimizer as opt           # noqa: E402
from core import gpu_backend as gb          # noqa: E402

_spec = importlib.util.spec_from_file_location("proto_rfe_gram",
                                               _ROOT / "dev" / "proto_rfe_gram.py")
proto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proto)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

STREAM_ENV = ("PHEASY_RFE_GRAM_CV", "PHEASY_RFE_GRAM_GB", "PHEASY_RFE_GRAM_STORE",
              "PHEASY_RFE_GRAM_DTYPE", "PHEASY_RFE_GRAM_TOTAL", "PHEASY_RFE_GRAM_ROUND_BUFFERS",
              "PHEASY_RFE_GRAM_GPU", "PHEASY_RFE_GRAM_DIR", "PHEASY_RFE_GRAM_KEEP",
              "PHEASY_RFE_GRAM_F32_MAXERR", "PHEASY_RFE_GRAM_GPU_DEVICES",
              "PHEASY_RFE_GRAM_GPU_GB", "PHEASY_RFE_GRAM_GPU_PANEL", "PHEASY_CV_FOLD_MODE",
              "PHEASY_RFE_GRAM_BLOCK_ROWS", "PHEASY_RFE_GRAM_GPU_DTYPE")


def small_problem(seed=1, n_cfg=40, rows_per=30, m=300, p=120, noise=0.01, dtype=np.float32):
    r = np.random.default_rng(seed)
    n = n_cfg * rows_per
    prime = sp.random(n, m, density=0.05, random_state=r, format="csr").astype(dtype)
    ns = (sp.random(m, p, density=0.03, random_state=r, format="csr")
          + sp.eye(m, p, format="csr")).tocsr()
    A = opt.TwoLevelSM(prime, ns)
    y = np.asarray(A @ r.standard_normal(p), dtype=np.float64)
    y = y + noise * np.std(y) * r.standard_normal(n)
    return A, y, rows_per


class _EnvCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rfe_gram_stream_")
        self.addCleanup(self.tmp.cleanup)
        self.store_dir = os.path.join(self.tmp.name, "store")
        clean = {k: v for k, v in os.environ.items() if k not in STREAM_ENV}
        clean.update(PHEASY_RFE_GRAM_DIR=self.store_dir, PHEASY_RFE_GRAM_GB="2",
                     PHEASY_GPU_MODE="cpu", PHEASY_GPU_SM="0")
        self.env = patch.dict(os.environ, clean, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def engine(self, A, y, splits, quiet=True, **env):
        with patch.dict(os.environ, {k: str(v) for k, v in env.items()}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                eng = opt._RFEStreamGramCV(A, y, np.arange(A.shape[1]), splits,
                                           verbose=not quiet)
            self.last_log = buf.getvalue()
            return eng

    def store_files(self):
        return sorted(os.listdir(self.store_dir)) if os.path.isdir(self.store_dir) else []


class EngineMatchesInRamEngine(_EnvCase):
    """Same identities as _RFEGramCV, in every storage layout."""

    def setUp(self):
        super().setUp()
        self.A, self.y, g = small_problem()
        self.splits = opt._make_cv_splits(self.A.shape[0], 5, 0, g)
        self.ref = opt._RFEGramCV(self.A, self.y, np.arange(self.A.shape[1]), self.splits)
        r = np.random.default_rng(7)
        self.loc = np.sort(r.choice(self.A.shape[1], 90, replace=False))

    def _assert_round(self, eng, loc, rtol=1e-11):
        want, got = self.ref.round(loc), eng.round(loc)
        np.testing.assert_allclose(got[0], want[0], rtol=0, atol=rtol * np.abs(want[0]).max())
        self.assertAlmostEqual(got[1] / want[1], 1.0, delta=rtol)
        self.assertAlmostEqual(got[2] / want[2], 1.0, delta=1e-9)
        np.testing.assert_allclose(eng.last_fold_rmse, self.ref.last_fold_rmse, rtol=rtol)
        ic_want = self.ref.round(loc, want_cv=False)
        ic_got = eng.round(loc, want_cv=False)
        self.assertAlmostEqual(ic_got[3] / ic_want[3], 1.0, delta=1e-9)

    def test_every_layout(self):
        for store, total, nbuf in itertools.product(("ram", "disk"), ("0", "1"), ("1", "2")):
            with self.subTest(store=store, total=total, nbuf=nbuf):
                eng = self.engine(self.A, self.y, self.splits, PHEASY_RFE_GRAM_STORE=store,
                                  PHEASY_RFE_GRAM_TOTAL=total,
                                  PHEASY_RFE_GRAM_ROUND_BUFFERS=nbuf)
                try:
                    self._assert_round(eng, self.loc)
                    self._assert_round(eng, np.arange(self.A.shape[1]))
                    np.testing.assert_allclose(eng.col_norms(),
                                               np.sqrt(self.ref.Gt.diagonal()), rtol=1e-13)
                    self.assertEqual(len(self.store_files()),
                                     0 if store == "ram" else 5 + int(total == "1"))
                finally:
                    eng.release()
                self.assertEqual(self.store_files(), [])

    def test_ridge_and_final_contract(self):
        # ranking / fold fits use ridge_alpha; the delivered final refit is
        # unregularized unless PHEASY_RFE_FINAL_RIDGE (same as _RFEGramCV)
        ref = opt._RFEGramCV(self.A, self.y, np.arange(self.A.shape[1]), self.splits,
                             ridge_alpha=0.3)
        with patch.dict(os.environ, {"PHEASY_RFE_GRAM_STORE": "ram"}):
            with contextlib.redirect_stdout(io.StringIO()):
                eng = opt._RFEStreamGramCV(self.A, self.y, np.arange(self.A.shape[1]),
                                           self.splits, ridge_alpha=0.3, verbose=False)
        try:
            for final in (False, True):
                w = ref.round(self.loc, want_cv=not final, final=final)
                g = eng.round(self.loc, want_cv=not final, final=final)
                np.testing.assert_allclose(g[0], w[0], rtol=0, atol=1e-11 * np.abs(w[0]).max())
        finally:
            eng.release()

    def test_scattered_rows_and_other_inputs(self):
        # interleaved folds and ungrouped KFold give non-contiguous fold rows;
        # CSR and dense inputs take other row-source branches
        A64 = opt._materialize_columns(self.A, np.arange(self.A.shape[1])).astype(np.float64)
        cases = [("interleaved", self.A, {"PHEASY_CV_FOLD_MODE": "interleaved"}, 30),
                 ("ungrouped", self.A, {}, None),
                 ("csr", sp.csr_matrix(A64), {}, 30),
                 ("dense", A64.toarray(), {}, 30)]
        for name, A, env, g in cases:
            with self.subTest(name), patch.dict(os.environ, env):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    splits = opt._make_cv_splits(A.shape[0], 5, 0, g)
                ref = opt._RFEGramCV(A, self.y, np.arange(A.shape[1]), splits)
                eng = self.engine(A, self.y, splits, PHEASY_RFE_GRAM_STORE="disk",
                                  PHEASY_RFE_GRAM_BLOCK_ROWS="97")
                try:
                    w, gt = ref.round(self.loc), eng.round(self.loc)
                    np.testing.assert_allclose(gt[0], w[0], rtol=0,
                                               atol=1e-11 * np.abs(w[0]).max())
                    self.assertAlmostEqual(gt[1] / w[1], 1.0, delta=1e-11)
                finally:
                    eng.release()


class Guards(_EnvCase):
    def setUp(self):
        super().setUp()
        self.A, self.y, g = small_problem()
        self.splits = opt._make_cv_splits(self.A.shape[0], 5, 0, g)

    def test_plan_refuses_before_allocating(self):
        with self.assertRaisesRegex(MemoryError, "do not fit: RAM budget"):
            self.engine(self.A, self.y, self.splits, PHEASY_RFE_GRAM_GB="0.0001",
                        PHEASY_RFE_GRAM_STORE="ram")
        self.assertFalse(os.path.isdir(self.store_dir), "a refused plan created the store")

    def test_plan_arithmetic_at_production_size(self):
        # c3=4.5: p = 77,063, K = 5.  float64 fold Grams are 47.5 GB each.
        with patch.object(opt, "_rfe_stream_threads", return_value=8), \
                patch.dict(os.environ, {"PHEASY_RFE_GRAM_STORE": "disk"}), \
                patch("shutil.disk_usage",
                      return_value=__import__("shutil")._ntuple_diskusage(0, 0, int(296e9))):
            plan = opt._rfe_stream_plan(1040112, 77063, 5, 200e9, self.store_dir)
        g64 = 8.0 * 77063 ** 2
        self.assertEqual(plan["store"], "disk")
        self.assertFalse(plan["total"])          # 6 x 47.5 GB does not fit 296 GB
        self.assertEqual(plan["nbuf"], 2)
        self.assertAlmostEqual(plan["store_bytes"] / 1e9, 5 * g64 / 1e9, places=3)
        self.assertGreater(plan["host_peak"], 2 * g64)
        self.assertLess(plan["host_peak"], 2 * g64 + 6e9)
        with patch.dict(os.environ, {"PHEASY_RFE_GRAM_STORE": "ram"}), \
                self.assertRaisesRegex(MemoryError, "do not fit"):
            opt._rfe_stream_plan(1040112, 77063, 5, 200e9, self.store_dir)

    def test_round_check_counts_active_page_cache_as_reclaimable(self):
        # A memory-limited job whose disk store reads filled the cgroup with
        # ACTIVE page cache: usage at the limit, almost no inactive_file.  The
        # default accounting (usage - inactive_file) saw ~2 kB free and the
        # round check raised MemoryError -- after the whole accumulation.
        eng = self.engine(self.A, self.y, self.splits, PHEASY_RFE_GRAM_STORE="disk")
        p = self.A.shape[1]
        root = os.path.join(self.tmp.name, "cg")
        os.makedirs(os.path.join(root, "job"))
        limit = 100_000_000

        def write(rel, text):
            with open(os.path.join(self.tmp.name, rel), "w") as fh:
                fh.write(text)
        write("cg/job/memory.max", str(limit))
        write("cg/job/memory.current", str(limit - 1000))
        write("cg/job/memory.stat", "anon 100000\nactive_file %d\ninactive_file 900\n"
              % (limit - 101000))
        write("self_cgroup", "0::/job\n")
        write("meminfo", "MemTotal: 1000000000 kB\nMemFree: 1000 kB\n"
                         "MemAvailable: 900000000 kB\n")
        try:
            with patch.object(opt, "_MEMINFO_PATH", os.path.join(self.tmp.name, "meminfo")), \
                    patch.object(opt, "_PROC_CGROUP_PATH",
                                 os.path.join(self.tmp.name, "self_cgroup")), \
                    patch.object(opt, "_CGROUP_ROOT", root):
                need = eng._round_bytes(p)
                self.assertLess(opt._available_memory_bytes(), need)
                self.assertGreater(opt._available_memory_bytes(reclaim_file_cache=True), need)
                eng.round(np.arange(p))           # must not raise
        finally:
            eng.release()

    def test_dropped_row_block_is_caught(self):
        real = opt._rfe_stream_blocks

        def drop_one(take, rows_k, blk):
            for i, item in enumerate(real(take, rows_k, blk)):
                if i == 1 and rows_k[0] == 0:
                    continue            # silently skip one block of fold 0
                yield item

        with patch.object(opt, "_rfe_stream_blocks", drop_one), \
                self.assertRaisesRegex(RuntimeError, "row accounting failed"):
            self.engine(self.A, self.y, self.splits, PHEASY_RFE_GRAM_BLOCK_ROWS="50",
                        PHEASY_RFE_GRAM_STORE="disk")
        self.assertEqual(self.store_files(), [], "a failed build left store files")

    def test_disk_round_trip_and_keep(self):
        eng = self.engine(self.A, self.y, self.splits, PHEASY_RFE_GRAM_STORE="disk",
                          PHEASY_RFE_GRAM_TOTAL="1", PHEASY_RFE_GRAM_KEEP="1")
        files = self.store_files()
        ref = opt._RFEGramCV(self.A, self.y, np.arange(self.A.shape[1]), self.splits)
        try:
            self.assertEqual(len(files), 6)
            for k in range(5):
                G = np.load(eng.store.paths[k], mmap_mode="r")
                self.assertTrue(G.flags.f_contiguous)
                np.testing.assert_array_equal(np.asarray(G), ref.G[k])
            np.testing.assert_array_equal(np.load(eng.store.paths[5]), ref.Gt)
        finally:
            eng.release()
        self.assertEqual(self.store_files(), files, "PHEASY_RFE_GRAM_KEEP=1 deleted the store")

    def test_float32_store_is_condition_checked(self):
        # high-SNR problem: the SSE cancellation guard refuses float32 storage
        eng = self.engine(self.A, self.y, self.splits, PHEASY_RFE_GRAM_DTYPE="float32",
                          PHEASY_RFE_GRAM_STORE="ram")
        try:
            with self.assertRaisesRegex(RuntimeError, "cannot resolve the validation error"):
                eng.round(np.arange(self.A.shape[1]))
            with patch.dict(os.environ, {"PHEASY_RFE_GRAM_F32_MAXERR": "1e-12"}), \
                    self.assertRaisesRegex(RuntimeError, "too coarse"):
                eng.round(np.arange(self.A.shape[1]), want_cv=False)
        finally:
            eng.release()

    def test_float32_store_within_its_estimate_on_low_snr(self):
        A, y, g = small_problem(noise=2.0)
        splits = opt._make_cv_splits(A.shape[0], 5, 0, g)
        ref = opt._RFEGramCV(A, y, np.arange(A.shape[1]), splits)
        eng = self.engine(A, y, splits, PHEASY_RFE_GRAM_DTYPE="float32",
                          PHEASY_RFE_GRAM_STORE="ram")
        try:
            loc = np.arange(A.shape[1])
            w, gt = ref.round(loc), eng.round(loc)
            est = 0.5 * opt._F32_UNIT * eng.max_sse_amplification
            self.assertLess(abs(gt[1] / w[1] - 1.0), 10 * est + 1e-9)
            self.assertLess(abs(gt[1] / w[1] - 1.0), 1e-4)
        finally:
            eng.release()


@unittest.skipUnless(HAVE_TORCH, "torch not importable")
class GpuAccumulator(_EnvCase):
    def test_panels_and_waves_are_exact(self):
        r = np.random.default_rng(3)
        p, n, blk = 300, 3000, 400
        X = r.standard_normal((n, p))
        want = X.T @ X
        fixed = blk * p * 8 + (256 << 20)
        for extra, waves in ((2e6, 1), (2.2e5, 2)):
            with self.subTest(waves=waves), patch.dict(os.environ, {
                    "PHEASY_RFE_GRAM_GPU_DEVICES": "cpu,cpu",
                    "PHEASY_RFE_GRAM_GPU_PANEL": "64",
                    "PHEASY_RFE_GRAM_GPU_GB": "%.9f" % ((fixed + extra) / 1e9)}):
                acc = gb.GpuGramAccumulator(p, block_rows=blk)
                self.assertEqual(acc.n_waves, waves)
                G = np.zeros((p, p), order="F")
                for w in range(acc.n_waves):
                    acc.begin_wave(w, G)
                    for i0 in range(0, n, blk):
                        acc.add(X[i0:i0 + blk])
                    acc.end_wave()
                acc.close()
                U = np.triu(G)
                np.testing.assert_allclose(U + np.triu(U, 1).T, want, rtol=0,
                                           atol=1e-13 * np.abs(want).max())

    def test_panel_that_fits_nowhere_raises(self):
        with patch.dict(os.environ, {"PHEASY_RFE_GRAM_GPU_DEVICES": "cpu",
                                     "PHEASY_RFE_GRAM_GPU_GB": "0.001"}), \
                self.assertRaisesRegex(MemoryError, "needs"):
            gb.GpuGramAccumulator(1000, block_rows=100)

    def test_float32_gpu_accumulation_is_guarded(self):
        # PHEASY_RFE_GRAM_GPU_DTYPE=float32 keeps a float64 STORE, so a guard
        # keyed on the storage dtype alone let it run unchecked
        A, y, g = small_problem()
        splits = opt._make_cv_splits(A.shape[0], 5, 0, g)
        loc = np.arange(A.shape[1])
        eng = self.engine(A, y, splits, PHEASY_RFE_GRAM_STORE="ram", PHEASY_RFE_GRAM_GPU="1",
                          PHEASY_RFE_GRAM_GPU_DTYPE="float32",
                          PHEASY_RFE_GRAM_GPU_DEVICES="cpu", PHEASY_RFE_GRAM_GPU_GB="1")
        ref = opt._RFEGramCV(A, y, loc, splits)
        try:
            self.assertEqual(eng.dtype, np.float64)
            self.assertIn("float32 GPU accumulation", self.last_log)
            self.assertGreater(eng.entry_unit, opt._F32_UNIT)
            with self.assertRaisesRegex(RuntimeError, "float32 GPU accumulation"):
                eng.round(loc)
            # accepted on purpose: it runs, and the real error is inside the estimate
            with patch.dict(os.environ, {"PHEASY_RFE_GRAM_F32_MAXERR": "1"}):
                got = eng.round(loc)
            want = ref.round(loc)
            est = 0.5 * eng.entry_unit * eng.max_sse_amplification
            self.assertLess(abs(got[1] / want[1] - 1.0), est)
        finally:
            eng.release()

    def test_engine_gpu_matches_host_and_does_not_fall_back(self):
        A, y, g = small_problem()
        splits = opt._make_cv_splits(A.shape[0], 5, 0, g)
        loc = np.arange(A.shape[1])
        host = self.engine(A, y, splits, PHEASY_RFE_GRAM_STORE="ram")
        gpu = self.engine(A, y, splits, PHEASY_RFE_GRAM_STORE="ram", PHEASY_RFE_GRAM_GPU="1",
                          PHEASY_RFE_GRAM_GPU_DEVICES="cpu", PHEASY_RFE_GRAM_GPU_GB="1",
                          PHEASY_RFE_GRAM_GPU_PANEL="32")
        try:
            self.assertIn("GPU float64 GEMM on cpu", self.last_log)
            self.assertEqual(gpu.entry_unit, 0.0)      # float64 throughout: no guard
            h, gg = host.round(loc), gpu.round(loc)
            np.testing.assert_allclose(gg[0], h[0], rtol=0, atol=1e-11 * np.abs(h[0]).max())
            self.assertAlmostEqual(gg[1] / h[1], 1.0, delta=1e-11)
        finally:
            host.release()
            gpu.release()
        # requested GPU, none usable: raise, never a quiet host dsyrk
        import torch
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                self.engine(A, y, splits, PHEASY_RFE_GRAM_GPU="1")


class RfeFitIntegration(_EnvCase):
    """PheasyRFECV end to end: the Phase-0 gate at test size."""

    @classmethod
    def setUpClass(cls):
        cls.problem = proto.make_problem(seed=0, n_cfg=40, rows_per=30, p2=40, p3=160)

    def test_streaming_reproduces_in_ram_engine_and_lsmr(self):
        A, y, protected, g = self.problem
        modes = ["lsmr", "gram_ram", "stream_ram", "stream_disk", "stream_1buf"]
        if HAVE_TORCH:
            modes.append("stream_gpu")
        results = {m: proto.run(m, A, y, protected, g) for m in modes}
        ok, ref, rows = proto.gate(results, 1e-6)
        detail = {m: c for m, _r, c, _p in rows}
        self.assertTrue(ok, detail)
        self.assertEqual(ref, "gram_ram")
        self.assertGreaterEqual(len(results["gram_ram"]["rounds"]), 5)
        for m in modes:
            meta = results[m]["meta"]
            if m.startswith("stream"):
                self.assertEqual(meta["gram_engine"]["mode"], "stream")
                self.assertEqual(meta["subset_solver"], "cpu_gram_exact")
                self.assertIn("[RFE-GRAM-STREAM]", results[m]["log"])
            else:
                self.assertNotIn("[RFE-GRAM-STREAM]", results[m]["log"])
        self.assertEqual(results["lsmr"]["meta"]["subset_solver"], "cpu")
        self.assertEqual(results["gram_ram"]["meta"]["gram_engine"]["mode"], "in_ram")

    def test_gate_flags_a_wrong_engine(self):
        A, y, protected, g = self.problem
        ref = proto.run("gram_ram", A, y, protected, g)
        bad = dict(ref, coef=ref["coef"] * (1 + 1e-4))
        ok, _ref, _rows = proto.gate({"gram_ram": ref, "bad": bad}, 1e-6)
        self.assertFalse(ok)

    def test_streaming_failure_raises_instead_of_falling_back(self):
        A, y, protected, g = self.problem
        with self.assertRaisesRegex(MemoryError, "do not fit"):
            proto.run("stream_ram", A, y, protected, g,
                      extra_env={"PHEASY_RFE_GRAM_GB": "0.00001"})

    def test_required_gpu_mode_with_host_streaming(self):
        # the host engine is an explicit exact solver, as the in-RAM engine is
        # under required mode; nothing may try the resident GPU operator
        A, y, protected, g = self.problem
        res = proto.run("stream_ram", A, y, protected, g,
                        extra_env={"PHEASY_GPU_MODE": "required"})
        self.assertEqual(res["meta"]["gram_engine"]["mode"], "stream")
        self.assertIsNone(res["meta"]["resident_fallback_reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
