import numpy as np, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACK = "/home/wangchao/software/AutoZT/tmp/modes/pack"
OUT = "/home/wangchao/software/AutoZT/tmp/modes/fig"
os.makedirs(OUT, exist_ok=True)

CASES = [("3.50", "3.5 A"), ("3.00", "3.0 A")]

for cut, label in CASES:
    d = np.load(os.path.join(PACK, "scattering_modes_c3_%s.npz" % cut))
    freq = np.asarray(d["frequency"])          # (nq, nband) THz
    gamma = np.asarray(d["gamma"])             # (nq, nband) THz
    giso = np.asarray(d["gamma_isotope"])      # (nq, nband) THz
    qp = np.asarray(d["qpoint"])
    gv = np.asarray(d["group_velocity"])       # (nq, nband, 3) THz*A = A/ps
    vnorm = np.linalg.norm(gv, axis=2)

    nq, nb = freq.shape
    tau_ps = 1.0 / (2.0 * np.maximum(gamma, 1e-30))
    tau_inv = 2.0 * gamma
    giso_inv = 2.0 * giso
    mfp_nm = vnorm * tau_ps / 10.0

    qnorm = np.linalg.norm(qp, axis=1)
    mask = np.zeros_like(freq, dtype=bool)
    mask[qnorm > 1e-8, :] = True
    mask &= freq > 0.10

    f = freq[mask]; ti = tau_inv[mask]; gi = giso_inv[mask]; mf = mfp_nm[mask]; vv = vnorm[mask]
    print("%s: %d modes used of %d" % (label, f.size, freq.size), flush=True)

    # masks for plotting only
    m_iso = gi > 1e-8                  # drop modes whose isotope rate is numerically zero
    m_v   = vv > 0.1                   # drop modes with vanishing group velocity (flat bands)""

    with open(os.path.join(OUT, "scattering_modes_c3_%s.csv" % cut), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frequency_THz", "tau_inv_phph_THz", "tau_ps", "tau_inv_isotope_THz",
                    "group_velocity_A_per_ps", "mfp_nm", "is_gamma_point", "band_index"])
        for iq in range(nq):
            for ib in range(nb):
                w.writerow(["%.6f" % freq[iq, ib], "%.6e" % tau_inv[iq, ib], "%.6e" % tau_ps[iq, ib],
                            "%.6e" % giso_inv[iq, ib], "%.6f" % vnorm[iq, ib],
                            "%.6f" % mfp_nm[iq, ib], int(qnorm[iq] <= 1e-8), ib])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ax = axes[0]
    ax.scatter(f, ti, s=6, alpha=0.45, c="#4C72B0", linewidths=0)
    ax.set_yscale("log"); ax.set_xlim(0, freq.max() * 1.02)
    ax.set_xlabel("frequency (THz)"); ax.set_ylabel(r"3-phonon rate $\tau^{-1}$ (THz)")
    ax.set_title("(a) three-phonon scattering rate"); ax.grid(True, which="both", ls=":", alpha=0.4)

    ax = axes[1]
    ax.scatter(f[m_iso], gi[m_iso], s=6, alpha=0.45, c="#DD8452", linewidths=0)
    sel = m_iso & (f > 2.0) & (f < 6.0)     # low-frequency window for the omega^4 guide
    if sel.any():
        A = np.median(gi[sel] / f[sel] ** 4)
        w_ref = np.linspace(1.5, 8.0, 200)
        ax.plot(w_ref, A * w_ref ** 4, "k--", lw=1.2, label=r"$\propto \omega^4$ (guide, 2-6 THz)")
        ax.legend(loc="lower right", fontsize=9)
    ax.set_yscale("log"); ax.set_xlim(0, freq.max() * 1.02)
    ax.set_xlabel("frequency (THz)"); ax.set_ylabel(r"isotope rate $\tau^{-1}_{\rm iso}$ (THz)")
    ax.set_title("(b) natural-isotope scattering rate"); ax.grid(True, which="both", ls=":", alpha=0.4)

    ax = axes[2]
    ax.scatter(f[m_v], mf[m_v], s=6, alpha=0.45, c="#55A868", linewidths=0)
    ax.set_yscale("log"); ax.set_xlim(0, freq.max() * 1.02)
    ax.set_xlabel("frequency (THz)"); ax.set_ylabel("mean free path (nm)")
    ax.set_title("(c) phonon mean free path"); ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle("Mg4C60: mode-resolved phonon scattering (%s force constants, 300 K, 8x8x8, isotope)" % label,
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "fig_scattering_c3_%s.png" % cut), dpi=160)
    fig.savefig(os.path.join(OUT, "fig_scattering_c3_%s.pdf" % cut))
    plt.close(fig)
    print("%s: median tau %.3f ps ; median mfp %.2f nm ; max mfp %.1f nm"
          % (label, np.median(tau_ps[mask]), np.median(mfp_nm[mask]), np.max(mfp_nm[mask])), flush=True)
print("done", flush=True)
