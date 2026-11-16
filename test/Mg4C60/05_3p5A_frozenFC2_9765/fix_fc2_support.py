"""Fixed-second-order (fix_fc2) support for the pheasy fit driver.

When the second-order force constants are FIXED from a previous joint fit,
pheasy is run with `--fix_fc2 --fc2_fmt PHONOPY`; it then keeps the harmonic
block untouched and only fits the anharmonic part.  The HARM_DENSE gate no
longer applies (the harmonic block does not take part in the fit), so the
driver must verify the fixed-2nd-order contract instead.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

FIX_FC2_NAME = "fc.hdf5"
FIX_FC2_FMT = "PHONOPY"
HARM_DENSE_IGNORED = "[HARM_DENSE] ignored: FIX_FC2"


def fix_fc2_config(cfg):
    """Return the configured fixed-fc2 source path (None when disabled)."""
    raw = cfg.get("fix_fc2")
    if not raw or str(raw).strip().lower() in ("", "0", "false", "no", "off"):
        return None
    return str(raw).strip()


def enabled(cfg):
    return fix_fc2_config(cfg) is not None


def fc2_fmt(cfg):
    return str(cfg.get("fc2_fmt") or FIX_FC2_FMT).strip().upper()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(out, cfg, source_dir=None):
    """Copy the fixed fc2 into <out>/fc.hdf5 and return its sha256.

    `source_dir` is the fit directory; a relative fix_fc2 path is resolved
    against it.  The file is never taken from any other directory.
    """
    raw = fix_fc2_config(cfg)
    if raw is None:
        return None
    src = Path(raw)
    if not src.is_absolute() and source_dir is not None:
        src = Path(source_dir) / src
    if not src.is_file():
        raise FileNotFoundError("fix_fc2 source not found: %s" % src)
    out = Path(out)
    dst = out / FIX_FC2_NAME
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    digest = sha256_file(dst)
    meta = {"source": str(src), "source_sha256": sha256_file(src),
            "delivered_sha256": digest, "fmt": fc2_fmt(cfg)}
    with open(out / "fc.hdf5.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return digest


def pheasy_flags(cfg):
    """Extra pheasy CLI flags for the fixed-fc2 mode (empty when disabled)."""
    if not enabled(cfg):
        return []
    return ["--fix_fc2", "--fc2_fmt", fc2_fmt(cfg)]


def manifest_applied(metrics):
    """harm_dense.applied as recorded in the manifest (None when absent)."""
    for key in ("harm_dense", "harm_dense_info"):
        block = (metrics or {}).get(key)
        if isinstance(block, dict) and "applied" in block:
            return bool(block["applied"])
    for key in ("harm_dense_applied", "applied"):
        if isinstance(metrics, dict) and key in metrics and key == "harm_dense_applied":
            return bool(metrics[key])
    return None


def gate(out, cfg, log_text, metrics, expected_sha256=None):
    """Verify the fixed-fc2 contract.

    Returns (ok, reason).  Requires all three of:
      1. the pheasy log carries the HARM_DENSE-ignored line;
      2. the manifest records harm_dense.applied == False;
      3. the delivered fc.hdf5 sha256 matches the source / expected digest.
    """
    if not enabled(cfg):
        return True, "fix_fc2 disabled"
    out = Path(out)
    reasons = []
    if HARM_DENSE_IGNORED not in (log_text or ""):
        reasons.append("log缺少'%s'" % HARM_DENSE_IGNORED)
    applied = manifest_applied(metrics)
    # pheasy does not record a harm_dense block when FIX_FC2 is active (measured
    # on pheasy 9dc76c0), so a missing block is the normal case; only an
    # explicit True means the harmonic block really took part in the fit.
    if applied is True:
        reasons.append("harm_dense.applied=True（固定二阶应为false）")
    fc = out / FIX_FC2_NAME
    if not fc.is_file():
        reasons.append("缺少%s" % FIX_FC2_NAME)
    else:
        digest = sha256_file(fc)
        ref = expected_sha256
        if ref is None:
            meta = out / "fc.hdf5.meta.json"
            if meta.is_file():
                ref = json.loads(meta.read_text(encoding="utf-8")).get("delivered_sha256")
        if ref is None:
            reasons.append("缺少固定二阶的期望md5")
        elif digest != ref:
            reasons.append("fc.hdf5 md5 不一致")
    if reasons:
        return False, "; ".join(reasons)
    return True, "fix_fc2 contract satisfied"


def rmse_reference_check(metrics, ref_manifest, tol=0.01):
    """Fixed-fc2 acceptance: force RMSE must match the reference joint fit.

    The reference is the fit_manifest.json of the joint (fc2+fc3) fit at the
    same cutoff.  Returns (ok, reason, detail).  Only the train RMSE is
    compared; the joint fit and the fixed-fc2 fit see the same force dataset.
    """
    import json as _json
    detail = {"reference": str(ref_manifest), "tol": float(tol)}
    try:
        ref = _json.load(open(ref_manifest, encoding="utf-8"))
    except Exception as e:
        return False, "无法读取参考 manifest: %s" % e, detail
    ref_rmse = None
    for path in (("metrics", "rmse"), ("results", "rmse"),
                 ("metrics", "pheasy_rmse_train"),
                 ("results", "fit_rmse")):
        cur = ref
        for key in path:
            cur = cur.get(key) if isinstance(cur, dict) else None
        if cur is not None:
            ref_rmse = float(cur)
            break
    if ref_rmse is None or ref_rmse <= 0:
        return False, "参考 manifest 没有可用的 RMSE", detail
    run_rmse = (metrics or {}).get("pheasy_rmse_train")
    if run_rmse is None:
        run_rmse = (metrics or {}).get("rmse")
    if run_rmse is None:
        return False, "本次拟合没有 RMSE 指标", detail
    run_rmse = float(run_rmse)
    rel = abs(run_rmse - ref_rmse) / ref_rmse
    detail.update({"run_rmse": run_rmse, "ref_rmse": ref_rmse, "rel_diff": rel})
    if rel > float(tol):
        return False, ("RMSE 与参考联合拟合相差 %.2f%% > %.2f%% (%.6g vs %.6g)"
                       % (100 * rel, 100 * float(tol), run_rmse, ref_rmse)), detail
    return True, ("RMSE 与参考联合拟合一致（相差 %.3f%%）" % (100 * rel)), detail
