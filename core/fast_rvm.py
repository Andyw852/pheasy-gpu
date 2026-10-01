"""Fast marginal-likelihood RVM (sparse Bayesian learning).

Tipping & Faul, "Fast Marginal Likelihood Maximisation for Sparse Bayesian
Models", AISTATS 2003.  The batch ARD evidence loop starts with every
coefficient active and inverts a p x p matrix, so its first iteration is O(p^3).
This algorithm starts from the empty model and adds / re-estimates / deletes one
basis function at a time, maintaining the leave-one-out statistics
  S_i = phi_i^T C_{-i}^{-1} phi_i,  Q_i = phi_i^T C_{-i}^{-1} t
incrementally, so each step is O(k*p) (k = active-set size) and no p x p matrix
is ever formed or factorized.

Model: t = Phi w + eps, eps ~ N(0, beta^-1 I), w_i ~ N(0, alpha_i^-1).
  l(alpha_i) = 0.5 [ ln alpha_i - ln(alpha_i + S_i) + Q_i^2/(alpha_i + S_i) ]
maximized at alpha_i = S_i^2/(Q_i^2 - S_i) when Q_i^2 > S_i, and at
alpha_i = infinity (delete) otherwise, with
  l_max = 0.5 [ Q_i^2/S_i - 1 - ln(Q_i^2/S_i) ].
"""
import numpy as np

_TINY = float(np.finfo(np.float64).tiny)


def _chol(A):
    """R with R R^T = A (Cholesky, eigen fallback)."""
    A = 0.5 * (A + A.T)
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(A)
        return evecs * np.sqrt(np.maximum(evals, 0.0))


def _refresh_sigma(G, S, alpha, beta):
    k = len(S)
    if k == 0:
        return np.zeros((0, 0))
    A = beta * np.asarray(G[np.ix_(S, S)], dtype=np.float64)
    A[np.diag_indices(k)] += np.array([alpha[int(i)] for i in S], dtype=np.float64)
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def _delete_sigma(Sigma, pos):
    """Inverse of Sigma with row/col pos removed (block-matrix identity)."""
    row = Sigma[pos, :].copy()
    sub = np.delete(np.delete(Sigma, pos, axis=0), pos, axis=1)
    rr = np.delete(row, pos)
    if Sigma[pos, pos] > _TINY and sub.size:
        sub = sub - np.outer(rr, rr) / Sigma[pos, pos]
    return sub


def _run_faml(G, b, diagG, n, beta, tol=1e-6, max_steps=None, add_batch=1,
              alpha_ceiling=1e12, refresh_every=0, verbose=False, free_idx=None,
              alpha_free=0.0):
    """One fast marginal-likelihood run at fixed beta (incremental updates).

    [HARM_DENSE] ``free_idx`` basis functions start IN the model with a flat
    prior (precision ``alpha_free``, default 0) and are never deleted or
    re-estimated, so the sparse search runs over the remaining (anharmonic)
    columns only -- the fast-RVM form of unpenalized covariates.
    """
    p = int(diagG.shape[0])
    S = np.empty(0, dtype=np.int64)
    alpha = {}
    v = np.empty(0, dtype=np.float64)
    Sigma = np.zeros((0, 0), dtype=np.float64)
    coef = np.zeros(p, dtype=np.float64)
    if max_steps is None:
        max_steps = max(1000, 20 * p)
    Sq = beta * diagG.copy()
    Qq = beta * b.copy()
    converged = False
    n_steps = 0
    refresh_every = int(refresh_every or 0)
    is_free = np.zeros(p, dtype=bool)

    def recompute_stats():
        k = len(S)
        if k == 0:
            return beta * diagG.copy(), beta * b.copy()
        R = _chol(Sigma)
        M = R.T @ np.asarray(G[S, :], dtype=np.float64)
        Rv = R.T @ v
        return (beta * diagG - beta * beta * np.einsum("ai,ai->i", M, M),
                beta * b - beta * beta * (M.T @ Rv))

    if free_idx is not None and len(free_idx):
        is_free[np.asarray(free_idx, dtype=np.int64)] = True
        S = np.asarray(np.flatnonzero(is_free), dtype=np.int64)
        alpha = {int(i): float(alpha_free) for i in S}
        Sigma = _refresh_sigma(G, S, alpha, beta)
        v = np.asarray(b[S], dtype=np.float64)
        Sq, Qq = recompute_stats()

    for step in range(int(max_steps)):
        n_steps = step + 1
        k = len(S)
        Sq = np.maximum(Sq, _TINY)
        q2 = Qq * Qq
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = q2 / Sq
            gain = np.where(q2 > Sq, 0.5 * (ratio - 1.0 - np.log(ratio)), 0.0)
        gain = np.where(np.isfinite(gain), gain, 0.0)
        act_gain = np.empty(k, dtype=np.float64)
        act_action = []
        for j in range(k):
            i = int(S[j])
            if is_free[i]:
                # flat prior: no delete, no re-estimate, never the best move
                act_gain[j] = -np.inf
                act_action.append(("fixed", None))
                gain[i] = -np.inf
                continue
            a = float(alpha[i])
            denom = a - Sq[i]
            if denom <= 1e-300:
                act_gain[j] = 0.0
                act_action.append(("delete", None))
            else:
                s_loo = a * Sq[i] / denom
                q_loo = a * Qq[i] / denom
                q2_loo = q_loo * q_loo
                l_old = 0.5 * (np.log(a) - np.log(a + s_loo) + q2_loo / (a + s_loo))
                if q2_loo <= s_loo:
                    act_gain[j] = -l_old
                    act_action.append(("delete", None))
                else:
                    a_new = s_loo * s_loo / (q2_loo - s_loo)
                    r_loo = q2_loo / s_loo
                    act_gain[j] = 0.5 * (r_loo - 1.0 - np.log(r_loo)) - l_old
                    act_action.append(("update", a_new))
            gain[i] = act_gain[j]
        best = int(np.argmax(gain))
        best_gain = float(gain[best])
        if not np.isfinite(best_gain) or best_gain <= tol:
            converged = True
            break
        hits = np.where(S == best)[0]
        pos = int(hits[0]) if hits.size else -1
        if pos >= 0:
            action, a_new = act_action[pos]
            if action == "delete":
                # Incremental downdate (block-inverse identity), O(k*p):
                #   Sq_j(S\k) = Sq_j(S) + beta^2 a_j^2 / Sigma_kk
                #   Qq_j(S\k) = Qq_j(S) + beta^2 a_j (r.v) / Sigma_kk
                # with a_j = (G[S,j])^T Sigma[:,k] and r = Sigma[:,k].
                r = Sigma[:, pos].copy()
                rv = float(r @ v)
                den = Sigma[pos, pos]
                if den > _TINY:
                    av = np.asarray(G[S, :], dtype=np.float64).T @ r
                    Sq = Sq + beta * beta * av * av / den
                    Qq = Qq + beta * beta * av * rv / den
                Sigma = _delete_sigma(Sigma, pos)
                S = np.delete(S, pos)
                v = np.delete(v, pos)
                del alpha[best]
            else:
                a_old = float(alpha[best])
                a_new = min(max(a_new, 1e-12), alpha_ceiling)
                d = a_new - a_old
                r = Sigma[:, pos].copy()
                rv = float(r @ v)
                den = 1.0 + d * Sigma[pos, pos]
                if abs(den) > _TINY:
                    av = np.asarray(G[S, :], dtype=np.float64).T @ r
                    Sq = Sq + beta * beta * d * av * av / den
                    Qq = Qq + beta * beta * d * av * rv / den
                    Sigma = Sigma - d * np.outer(r, r) / den
                    alpha[best] = a_new
        else:
            in_S = np.zeros(p, dtype=bool)
            in_S[S] = True
            cand = np.where((q2 > Sq) & (gain > tol) & ~in_S)[0]
            if cand.size == 0:
                converged = True
                break
            order = cand[np.argsort(gain[cand])[::-1]][:max(1, int(add_batch))]
            for i in order:
                i = int(i)
                if i in alpha:
                    continue
                # [FIX RVM-BETA] Sq/Qq were updated by the previous addition of
                # this batch; q2 from the top of the step is stale for i
                q2_i = float(Qq[i] * Qq[i])
                if q2_i <= Sq[i]:
                    continue
                a_i = float(Sq[i] * Sq[i] / (q2_i - Sq[i]))
                a_i = min(max(a_i, 1e-12), alpha_ceiling)
                sc = a_i + Sq[i]
                if sc <= _TINY:
                    continue
                c = beta * np.asarray(G[S, i], dtype=np.float64).ravel()
                r = Sigma @ c
                rv = float(r @ v) if v.size else 0.0
                if len(S):
                    e = (np.asarray(G[S, :], dtype=np.float64).T @ r
                         - np.asarray(G[i, :], dtype=np.float64).ravel())
                else:
                    e = -np.asarray(G[i, :], dtype=np.float64).ravel()
                Sq = Sq - beta * beta * e * e / sc
                Qq = Qq - beta * beta * e * (rv - b[i]) / sc
                kk = len(S)
                Sigma_new = np.empty((kk + 1, kk + 1), dtype=np.float64)
                Sigma_new[:kk, :kk] = Sigma + np.outer(r, r) / sc
                Sigma_new[:kk, kk] = -r / sc
                Sigma_new[kk, :kk] = -r / sc
                Sigma_new[kk, kk] = 1.0 / sc
                Sigma = Sigma_new
                S = np.append(S, i)
                v = np.append(v, b[i])
                alpha[i] = a_i
        if refresh_every and len(S) and (step + 1) % refresh_every == 0:
            Sigma = _refresh_sigma(G, S, alpha, beta)
            v = np.asarray(b[S], dtype=np.float64)
            Sq, Qq = recompute_stats()
        if verbose and (step % 50 == 0):
            print("[RVM] step %d: active=%d best_gain=%.3e beta=%.4e"
                  % (step, len(S), best_gain, beta), flush=True)
    if len(S):
        Sigma = _refresh_sigma(G, S, alpha, beta)
        v = np.asarray(b[S], dtype=np.float64)
        coef[S] = beta * (Sigma @ v)
    return {"coef": coef, "active": S,
            "alpha": np.array([alpha[int(i)] for i in S], dtype=np.float64),
            "beta": float(beta), "n_steps": int(n_steps),
            "converged": bool(converged)}


def _evidence_beta(G, run, n, rss):
    """(N - sum gamma) / rss for the active set of one fast-RVM run."""
    S = np.asarray(run["active"], dtype=np.int64)
    if S.size == 0:
        return n / rss
    a = np.asarray(run["alpha"], dtype=np.float64)
    A = run["beta"] * np.asarray(G[np.ix_(S, S)], dtype=np.float64)
    A[np.diag_indices(S.size)] += a
    try:
        L = np.linalg.cholesky(A)
        Linv = np.linalg.solve(L, np.eye(S.size))
        sig_diag = np.einsum("ij,ij->j", Linv, Linv)
    except np.linalg.LinAlgError:
        sig_diag = np.diag(np.linalg.pinv(A))
    gamma = 1.0 - a * sig_diag
    dof = float(n) - float(np.clip(gamma, 0.0, 1.0).sum())
    return max(dof, 1.0) / rss


def _prune_and_refit(G, b, active, alpha, beta, threshold, free_idx=None):
    """Zero the active coefficients whose precision exceeds threshold.

    ARDRegression prunes lambda >= threshold_lambda; the fast maximization
    keeps any finite alpha, so the same rule is applied at the end.  The
    retained coefficients are re-solved exactly on that support.  Free
    ([HARM_DENSE]) basis functions are always retained.
    """
    p = int(G.shape[0])
    keep = alpha < float(threshold)
    if free_idx is not None and len(free_idx):
        keep = keep | np.isin(np.asarray(active, dtype=np.int64),
                              np.asarray(free_idx, dtype=np.int64))
    S = np.asarray(active[keep], dtype=np.int64)
    coef = np.zeros(p, dtype=np.float64)
    if S.size:
        k = S.size
        A = beta * np.asarray(G[np.ix_(S, S)], dtype=np.float64)
        A[np.diag_indices(k)] += alpha[keep]
        try:
            w = beta * np.linalg.solve(A, np.asarray(b[S], dtype=np.float64))
        except np.linalg.LinAlgError:
            w = beta * (np.linalg.pinv(A) @ np.asarray(b[S], dtype=np.float64))
        coef[S] = w
    return coef, S, np.asarray(alpha[keep], dtype=np.float64)


def fast_rvm(G, b, yty, n_samples, y_var=None, beta=None, beta_iters=5,
             tol=1e-6, max_steps=None, add_batch=1, alpha_ceiling=1e12,
             prune_threshold=1e4, refresh_every=0, verbose=False, free=None,
             alpha_free=0.0):
    """Fit the sparse Bayesian regression model on the Gram (G, b).

    G = X^T X (p x p), b = X^T y, yty = y^T y, n_samples = rows of X.
    beta = 1/sigma^2 is the noise precision.  When beta is None it starts at
    1/var(y) and is re-estimated from the residual, re-running the fast
    maximization, for beta_iters rounds.

    free ([HARM_DENSE], bool mask): basis functions kept in the model with a
    flat prior -- never deleted, re-estimated or pruned.

    Returns a dict with coef, active, alpha, beta and diagnostics.
    """
    G = np.asarray(G, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    n = int(n_samples)
    yty = float(yty)
    diagG = np.diag(G).copy()
    free_idx = None
    if free is not None:
        _fm = np.asarray(free, dtype=bool).ravel()
        if _fm.shape != diagG.shape:
            raise ValueError("fast_rvm free mask must have one entry per feature")
        free_idx = np.flatnonzero(_fm) if _fm.any() else None
    # [FIX RVM-BETA] a supplied beta (PHEASY_RVM_BETA) is FIXED, as documented;
    # it used to be only the starting value of the re-estimation loop
    fixed_beta = beta is not None
    if beta is None:
        v0 = y_var if y_var is not None else yty / max(n, 1)
        beta = 1.0 / max(float(v0), _TINY)
    last = None
    for it in range(1 if fixed_beta else max(1, int(beta_iters))):
        last = _run_faml(G, b, diagG, n, beta, tol=tol, max_steps=max_steps,
                         add_batch=add_batch, alpha_ceiling=alpha_ceiling,
                         refresh_every=refresh_every, verbose=verbose,
                         free_idx=free_idx, alpha_free=alpha_free)
        coef = last["coef"]
        rss = yty - 2.0 * float(coef @ b) + float(coef @ (G @ coef))
        rss = max(rss, _TINY)
        # [FIX RVM-BETA] evidence (ML-II) noise update, Tipping 2001 eq. (18) /
        # Tipping & Faul 2003: beta = (N - sum_i gamma_i) / ||t - Phi mu||^2,
        # gamma_i = 1 - alpha_i Sigma_ii.  n / rss ignored the well-determined
        # parameters, overestimating beta by ~N/(N - k) -> too little noise,
        # too many basis functions kept.
        if fixed_beta:
            break
        beta_new = _evidence_beta(G, last, n, rss)
        if verbose:
            print("[RVM] beta round %d: beta %.6e -> %.6e  active=%d rss=%.4e"
                  % (it, beta, beta_new, len(last["active"]), rss), flush=True)
        if abs(beta_new - beta) <= 1e-6 * beta:
            beta = beta_new
            last = _run_faml(G, b, diagG, n, beta, tol=tol, max_steps=max_steps,
                             add_batch=add_batch, alpha_ceiling=alpha_ceiling,
                             refresh_every=refresh_every, verbose=False,
                             free_idx=free_idx, alpha_free=alpha_free)
            break
        beta = beta_new
    last["beta"] = float(beta)
    if prune_threshold and prune_threshold > 0:
        coef, S, a = _prune_and_refit(G, b, last["active"], last["alpha"],
                                      last["beta"], prune_threshold,
                                      free_idx=free_idx)
        last["coef"] = coef
        last["active"] = S
        last["alpha"] = a
        last["pruned_to"] = int(S.size)
    return last
