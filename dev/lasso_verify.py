import numpy as np
from pheasy_gpu.core import gpu_backend as gb
from sklearn.linear_model import LassoCV

rng = np.random.default_rng(0)
A = rng.standard_normal((2000, 400))
beta = rng.standard_normal(400); beta[100:] = 0.0
y = A @ beta + 0.1 * rng.standard_normal(2000)
cn = np.sqrt((A*A).sum(axis=0)); cn = np.where(cn < 1e-30, 1.0, cn)
As = A / cn
alphas = np.logspace(-6, -1, 12)

lcv = LassoCV(alphas=alphas, cv=5, fit_intercept=False, tol=1e-4, max_iter=20000, random_state=0)
lcv.fit(As, y)
print("sklearn LassoCV: alpha=%.6e  nnz=%d" % (lcv.alpha_, int(np.count_nonzero(lcv.coef_))), flush=True)

gcv = gb.GpuLassoCV(alphas, cv=5, tol=1e-6, max_iter=20000, rand_seed=0)
gcv.fit(As, y)
print("gpu GpuLassoCV : alpha=%.6e  nnz=%d" % (gcv.alpha_, int(np.count_nonzero(gcv.coef_))), flush=True)
d = float(np.linalg.norm(lcv.coef_ - gcv.coef_) / max(np.linalg.norm(lcv.coef_), 1e-30))
print("coef rel diff = %.3e" % d, flush=True)
print("alpha rel diff = %.3e" % (abs(lcv.alpha_ - gcv.alpha_) / max(lcv.alpha_, 1e-30)), flush=True)
print("DONE", flush=True)

