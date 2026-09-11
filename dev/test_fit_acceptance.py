import unittest
import warnings
import numpy as np
import scipy.sparse as sp
from core.optimizer import Optimizer, TwoLevelSM

class FitAcceptanceTest(unittest.TestCase):
    def _make_twolevel(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((40, 6))
        x_true = rng.standard_normal(6)
        y = A @ x_true + 1e-4 * rng.standard_normal(40)
        sm = sp.csr_matrix(A)
        ns = sp.eye(6, format="csr")
        return TwoLevelSM(sm, ns), y

    def test_converged_ols_is_accepted(self):
        A, y = self._make_twolevel()
        opt = Optimizer("OLS", max_iter=5000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.fit(A, y)
        self.assertIn("fit_accepted", opt.results)
        self.assertIn("status", opt.results)
        self.assertTrue(opt.results["fit_accepted"])
        self.assertEqual(opt.results["status"], "fit_returned")

    def test_iteration_limited_ols_is_not_accepted(self):
        A, y = self._make_twolevel()
        opt = Optimizer("OLS", max_iter=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.fit(A, y)
        self.assertFalse(opt.results["fit_accepted"])
        self.assertEqual(opt.results["status"], "fit_returned_not_accepted")

if __name__ == "__main__":
    unittest.main()
