import os
import unittest
import warnings
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
from core.optimizer import Optimizer, TwoLevelSM

class JacobiSemanticsTest(unittest.TestCase):
    """Jacobi column-scaling must not change the converged OLS answer.

    This is the semantics-preservation contract: PHEASY_OLS_JACOBI=1 solves the
    right-preconditioned system (A D) z = y, then x = D z, so a converged fit
    must agree with the unscaled LSMR solution (and both with the dense SVD).
    """

    def _make_twolevel(self, n=60, p=8, seed=3):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, p))
        x_true = rng.standard_normal(p)
        y = A @ x_true
        sm = sp.csr_matrix(A)
        ns = sp.eye(p, format="csr")
        return TwoLevelSM(sm, ns), y, A, x_true

    def _solve(self, A, y, jacobi):
        opt = Optimizer("OLS", max_iter=2000)
        with patch.dict(os.environ, {"PHEASY_OLS_JACOBI": "1" if jacobi else "0"}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                opt.fit(A, y)
        return opt.results["coef"], opt.results

    def test_jacobi_preserves_converged_solution(self):
        A, y, Adense, x_true = self._make_twolevel()
        off, res_off = self._solve(A, y, False)
        on, res_on = self._solve(A, y, True)
        # SVD ground truth via normal equations on the dense matrix.
        ref = np.linalg.lstsq(Adense, y, rcond=None)[0]
        np.testing.assert_allclose(off, ref, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(on, ref, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(off, on, rtol=1e-6, atol=1e-6)
        self.assertTrue(res_off["fit_accepted"])
        self.assertTrue(res_on["fit_accepted"])

if __name__ == "__main__":
    unittest.main()
