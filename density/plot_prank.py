#!/usr/bin/env python3
"""
Grafici i ekstrapolacija iz geometricPrank_all.csv (kolone: definition,d,N,p,n_graphs,
mean_prk_over_N,stderr). Koristi se samo definicija A.

Pokretanje:   python3 plot_prank.py geometricPrank_all.csv [--out figs]

Izlaz u --out:
  fig1_vs_N.png        p-rk/N u odnosu na N (log), panel po p, linija po d, SEM error barovi
  fig2_extrapolation.png  isti podaci u odnosu na 1/N (gore) i 1/sqrt(N) (dole), tezinski
                          linearni fit; kvadrat na x=0 je ekstrapolirano c_p(d) ± greska
  extrapolation.csv    c_inf za oba modela, chi2/dof, i "sistematska" razlika izmedju modela
Ako je chi2/dof > 1, greska fita se mnozi sa sqrt(chi2/dof).
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wls(x, y, s):
    """Tezinski fit y = c + b*x. Vraca c, err(c), b, chi2/dof."""
    s = np.maximum(s, 1e-12)
    X = np.c_[np.ones_like(x), x]
    W = 1 / s**2
    C = np.linalg.inv(X.T @ (X * W[:, None]))
    b = C @ (X.T @ (W * y))
    chi2 = float((W * (y - X @ b) ** 2).sum())
    dof = len(x) - 2
    red = chi2 / dof if dof > 0 else np.nan
    err = np.sqrt(C[0, 0]) * (np.sqrt(red) if (dof > 0 and red > 1) else 1.0)
    return b[0], err, b[1], red


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.csv)
    df = df[df["definition"] == "A"].rename(columns={"mean_prk_over_N": "m", "stderr": "s"})
    ps = sorted(df["p"].unique())
    ds = sorted(df["d"].unique())
    cols = dict(zip(ds, plt.cm.viridis(np.linspace(0, 0.9, len(ds)))))
    models = {"1/N": 1.0, "1/sqrt(N)": 0.5}

    rows = []
    for p in ps:
        for d in ds:
            g = df[(df.p == p) & (df.d == d)].sort_values("N")
            N, y, s = g["N"].values.astype(float), g["m"].values, g["s"].values
            r = {"p": p, "d": d}
            for name, ex in models.items():
                c, e, b, red = wls(N**-ex, y, s)
                r[f"c_inf[{name}]"], r[f"err[{name}]"], r[f"chi2dof[{name}]"] = c, e, red
            r["syst"] = abs(r["c_inf[1/N]"] - r["c_inf[1/sqrt(N)]"])
            rows.append(r)
    fits = pd.DataFrame(rows)
    fits.to_csv(f"{a.out}/extrapolation.csv", index=False)

    # Slika 1: vs N
    fig, axes = plt.subplots(1, len(ps), figsize=(4.2 * len(ps), 4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, ps):
        for d in ds:
            g = df[(df.p == p) & (df.d == d)].sort_values("N")
            ax.errorbar(g["N"], g["m"], yerr=g["s"], marker="o", ms=3, capsize=2,
                        color=cols[d], label=f"d={d:g}")
        ax.set_xscale("log"); ax.set_title(f"p = {p}"); ax.set_xlabel("N"); ax.grid(alpha=.3)
    axes[0].set_ylabel("p-rk / N"); axes[0].legend(fontsize=8)
    fig.suptitle("p-rk/N u odnosu na N (greska = SEM)")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig1_vs_N.png", dpi=150); plt.close(fig)

    # Slika 2: ekstrapolacija, dva modela
    fig, axes = plt.subplots(2, len(ps), figsize=(4.2 * len(ps), 7.5), squeeze=False)
    for row, (name, ex) in enumerate(models.items()):
        for ax, p in zip(axes[row], ps):
            for d in ds:
                g = df[(df.p == p) & (df.d == d)].sort_values("N")
                N = g["N"].values.astype(float)
                x = N**-ex
                f = fits[(fits.p == p) & (fits.d == d)].iloc[0]
                c, e = f[f"c_inf[{name}]"], f[f"err[{name}]"]
                _, _, b, _ = wls(x, g["m"].values, g["s"].values)
                ax.errorbar(x, g["m"], yerr=g["s"], fmt="o", ms=3, capsize=2, color=cols[d],
                            label=f"d={d:g}")
                xs = np.array([0, x.max()])
                ax.plot(xs, c + b * xs, "--", lw=1, color=cols[d])
                ax.errorbar([0], [c], yerr=[e], fmt="s", ms=5, capsize=3, color=cols[d])
            ax.set_title(f"p = {p}   (x = {name})"); ax.set_xlabel(name); ax.grid(alpha=.3)
            ax.set_xlim(left=-0.01 * ax.get_xlim()[1])
        axes[row][0].set_ylabel("p-rk / N")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Tezinski linearni fit; kvadrat na x=0 je ekstrapolirano c_p(d)")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig2_extrapolation.png", dpi=150); plt.close(fig)

    # Tabela
    print("\nc_inf ± greska fita (chi2/dof) za oba modela; 'syst' = |razlika modela|")
    print(f"{'d':>5}{'p':>3} | {'1/N':^26} | {'1/sqrt(N)':^26} | {'syst':>7}")
    for _, r in fits.iterrows():
        print(f"{r.d:>5g}{int(r.p):>3} | "
              f"{r['c_inf[1/N]']:.4f}±{r['err[1/N]']:.4f} ({r['chi2dof[1/N]']:4.1f}) | "
              f"{r['c_inf[1/sqrt(N)]']:.4f}±{r['err[1/sqrt(N)]']:.4f} ({r['chi2dof[1/sqrt(N)]']:4.1f}) | "
              f"{r.syst:.4f}")
    print(f"\nSlike i extrapolation.csv su u '{a.out}/'")


if __name__ == "__main__":
    main()