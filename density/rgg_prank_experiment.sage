# -*- coding: utf-8 -*-
#
# geometricPrankV2.sage -- p-rank gustina RGG(N, d) na torusu + kontrola (slucajni regularni grafovi)
#
# Razlike u odnosu na geometricPrankFast.sage:
#  1. Izbacena definicija B (B = A + (k-1)/N je redundantna). p-rk = N - rank_p(L) - k.
#     Cuva se samo A, a k/N se i dalje belezi u izvestaju.
#  2. Stepeni: d in {3, 4, 5, 6, 8, 12}  (4.5 je zamenjeno sa 4 i 5, jer je 4.5 na pragu perkolacije).
#  3. Gusca mreza N sa korakom sqrt(2): 125 ... 4000, i vise replika za male N.
#  4. Dva modela fita, oba tezinska: c(N) = c_inf + a/N  i  c(N) = c_inf + a/sqrt(N).
#     Prijavljuju se c_inf +- greska i chi2/dof za oba, plus "syst" = |razlika modela|.
#  5. Kontrolni mod `control`: slucajni d-regularni (multi)grafovi, unija d nezavisnih
#     uniformnih savrsenih uparivanja (Meszaros). Ocekuje se p-rk/N -> 0 i raspodela p-ranka
#     (za neparno d) P(r) = p^(-r(r+1)/2) prod_{i>r}(1-p^-i) / prod_{i>=1}(1-p^(-2i)).
#  6. Jezgro racuna i sa visestrukim granama (tezinski Laplasijan), pa radi i za regularne multigrafove.
#
# Kao i ranije: 2-jezgro (skidanje listova ne menja sandpile grupu), gusta matrica nad GF(p)
# (LinBox/M4RI), vremenski budzet sa "talasima" replika.
#
# POKRETANJE (BLAS niti iskljuciti VAN Sage-a):
#   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
#   sage geometricPrankV2.sage test                      # provera ispravnosti (brzo)
#   sage geometricPrankV2.sage [FAKTOR] [JEZGRA] [BUDZET_s]
#   sage geometricPrankV2.sage control [R] [N] [JEZGRA] [BUDZET_s]    # default R=1000, N=1000
#
# Izlaz (prefiks geometricPrank2):
#   _all.csv  (kolone kao ranije, definition=A; kompatibilno sa plot_prank.py)
#   _d<d>.txt, _summary.txt, _total.txt, _control.txt

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import math
import sys
import time

import numpy as np
from multiprocessing import Pool
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# NAPOMENA o Sage preparseru: sufiks r (1r, 1.0r) daje cist Python int/float.

DEGREES = [float(x) for x in [3, 4, 5, 6, 8, 12]]
N_VALUES = [int(x) for x in [125, 177, 250, 354, 500, 707, 1000, 1414, 2000, 2828, 4000]]
NUM_GRAPHS_PER_N = {125: 2000, 177: 1500, 250: 1000, 354: 800, 500: 600, 707: 400,
                    1000: 300, 1414: 150, 2000: 100, 2828: 40, 4000: 20}
PRIMES = [int(x) for x in [2, 3, 5, 7]]
REPORT_EVERY = 100
SEED = 12345
TIME_BUDGET = float(1500)
OUT = "geometricPrank2"
MODELS = [("1/N", 1.0r), ("1/sqrt(N)", 0.5r)]


# Pomocne funkcije

def fmt(x):
    """|x| < 1 -> 1 znacajna cifra, inace 2 decimale (za progres)."""
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) < 1:
        return "%.1g" % x
    return "%.2f" % x


def fmt4(x):
    return "%.4f" % float(x)


def connectivity_threshold(N):
    N = int(N)
    return math.sqrt(math.log(N) / (math.pi * N))


def radius_for_degree(N, d):
    return math.sqrt(float(d) / (math.pi * int(N)))


def dense_modp(n, ent, p):
    """Gusta n x n matrica nad GF(p) iz dict-a {(i,j): v}."""
    M = matrix(GF(int(p)), int(n), int(n), ent, sparse=False)
    if M.is_sparse():
        M = M.dense_matrix()
    return M


# Generatori grafova (samo liste grana)

def rgg_pairs(N, r, rng):
    N = int(N)
    pts = rng.random((N, 2r))
    tree = cKDTree(pts, boxsize=1.0r)          # torus
    return tree.query_pairs(float(r), output_type="ndarray")


def regular_pairs(N, d, rng):
    """Unija d nezavisnih uniformnih savrsenih uparivanja na N (parnih) cvorova.
    Moze imati visestruke grane (multigraf), kao u Meszarosovom modelu."""
    N = int(N)
    out = []
    for _ in range(int(d)):
        perm = rng.permutation(N)
        out.append(np.stack([perm[0r::2r], perm[1r::2r]], axis=1))
    return np.concatenate(out, axis=0)


# Jezgro: 2-core + rank mod p po komponentama jezgra (tezinski Laplasijan)

def two_core_mask(A, N):
    """Maska cvorova koji ostaju posle skidanja cvorova (tezinskog) stepena <= 1."""
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    alive = np.ones(N, dtype=bool)
    while True:
        rm = alive & (deg <= 1r)
        if not rm.any():
            return alive
        alive &= ~rm
        deg = deg - A.dot(rm.astype(np.int64))


def core_components(A, alive):
    """Lista (n, rows, cols, vals, diag) po povezanoj komponenti 2-jezgra."""
    idx = np.flatnonzero(alive)
    if idx.size == 0r:
        return []
    Ac = A[idx][:, idx].tocsr()
    ncomp, lab = connected_components(Ac, directed=False)
    degc = np.asarray(Ac.sum(axis=1)).ravel().astype(np.int64)
    sizes = np.bincount(lab, minlength=ncomp)
    order = np.argsort(lab, kind="stable")
    starts = np.cumsum(sizes) - sizes
    pos = np.empty(idx.size, dtype=np.int64)
    pos[order] = np.arange(idx.size) - starts[lab[order]]

    Co = Ac.tocoo()
    ec = lab[Co.row]
    eorder = np.argsort(ec, kind="stable")
    ecnt = np.bincount(ec, minlength=ncomp)
    estarts = np.cumsum(ecnt) - ecnt

    comps = []
    for c in range(int(ncomp)):
        n = int(sizes[c])
        nodes = order[starts[c]:starts[c] + n]
        e = eorder[estarts[c]:estarts[c] + ecnt[c]]
        rows = pos[Co.row[e]].tolist()
        cols = pos[Co.col[e]].tolist()
        vals = Co.data[e].tolist()
        diag = degc[nodes].tolist()
        comps.append((n, rows, cols, vals, diag))
    return comps


def comp_prank(n, rows, cols, vals, diag, primes):
    """p-rk komponente = n - 1 - rank_p(L), L = D - A (A tezinska)."""
    out = {}
    for p in primes:
        ent = {(i, j): (-v) % p for i, j, v in zip(rows, cols, vals)}
        for i, dg in enumerate(diag):
            v = dg % p
            if v:
                ent[(i, i)] = v
        M = dense_modp(n, ent, p)
        out[p] = int(n - 1r - int(M.rank()))
    return out


def p_ranks_from_pairs(N, pairs, primes):
    """Vraca (pr, k, ncore): pr[p] = p-rk celog grafa, k = broj komponenti, ncore = |2-jezgro|."""
    N = int(N)
    m = int(len(pairs))
    pr = {p: 0r for p in primes}
    if m == 0r:
        return pr, N, 0r
    i = pairs[:, 0r]
    j = pairs[:, 1r]
    A = coo_matrix((np.ones(2r * m, dtype=np.int64),
                    (np.concatenate([i, j]), np.concatenate([j, i]))),
                   shape=(N, N)).tocsr()          # duplikati se sabiraju (multigraf)
    k, _ = connected_components(A, directed=False)
    alive = two_core_mask(A, N)
    for (n, rows, cols, vals, diag) in core_components(A, alive):
        res = comp_prank(n, rows, cols, vals, diag, primes)
        for p in primes:
            pr[p] += res[p]
    return pr, int(k), int(alive.sum())


# Radnik

DEADLINE = float("inf")
_TP = None


def _init_worker(deadline):
    global DEADLINE, _TP
    DEADLINE = deadline
    try:
        from threadpoolctl import threadpool_limits
        _TP = threadpool_limits(limits=1r)
    except Exception:
        pass


def process_one_graph(task):
    """task = (kind, d, N, seed); kind = 'rgg' ili 'reg'. None ako je budzet istekao."""
    kind, d, N, seed = task
    if time.time() > DEADLINE:
        return None
    t0 = time.time()
    N = int(N)
    rng = np.random.default_rng(seed)
    if kind == "rgg":
        d = float(d)
        r = radius_for_degree(N, d)
        pairs = rgg_pairs(N, r, rng)
    else:
        d = int(d)
        r = float("nan")
        pairs = regular_pairs(N, d, rng)
    pr, k, ncore = p_ranks_from_pairs(N, pairs, PRIMES)
    return kind, d, N, pr, k, r, ncore, time.time() - t0


# Eksperiment

def run_experiment(kind, degrees, N_values, num_graphs, num_cores, budget, verbose=True):
    ss = np.random.SeedSequence(int(SEED))
    tasks = []
    for d in degrees:
        for N in N_values:
            for idx in range(num_graphs[N]):
                key = float(0) if idx < 2 else idx / float(num_graphs[N])   # "talasi"
                tasks.append((key, -N, (kind, d, N, ss.spawn(1)[0])))
    tasks.sort(key=lambda t: (t[0], t[1]))
    tasks = [t[2] for t in tasks]
    total = len(tasks)

    print(f"[{kind}] 2-core + gusti GF(p) | d={degrees} | N={N_values} | p={PRIMES}")
    print(f"ukupno {total} grafova | jezgara: {num_cores} | budzet: {budget:.0f}s")
    print("-" * 78)

    raw = {d: {N: {p: [] for p in PRIMES} for N in N_values} for d in degrees}
    kk = {d: {N: [] for N in N_values} for d in degrees}
    tim = {N: [] for N in N_values}

    t0 = time.time()
    deadline = t0 + budget
    done = 0
    skipped = 0
    with Pool(num_cores, initializer=_init_worker, initargs=(deadline,)) as pool:
        for res in pool.imap_unordered(process_one_graph, tasks, chunksize=1):
            done += 1
            if res is None:
                skipped += 1
                continue
            _, d, N, pr, k, r, ncore, dt = res
            for p in PRIMES:
                raw[d][N][p].append(pr[p])
            kk[d][N].append(k)
            tim[N].append(dt)
            if verbose and done % REPORT_EVERY == 0:
                el = time.time() - t0
                eta = el / done * (total - done)
                rest = max(float(0), deadline - time.time())
                print(f"  [{done:>6}/{total}] d={d:<4} N={N:<5} k/N={fmt(k / float(N))} "
                      f"core/N={fmt(ncore / float(N))} {PRIMES[0]}-rk/N={fmt(pr[PRIMES[0]] / float(N))}"
                      f" | {el:.0f}s, ETA ~{min(eta, rest):.0f}s")
                sys.stdout.flush()
    if skipped:
        print(f"\n!!! Budzet istekao: preskoceno {skipped} od {total} grafova "
              f"(stvarni broj je u koloni n_graphs).")
    return raw, kk, tim


# Fit i izvestavanje

def wls(x, y, s):
    """Tezinski fit y = c + b x. Vraca c, err_c, b, err_b, chi2/dof.
    Ako je chi2/dof > 1, greske se mnoze sa sqrt(chi2/dof)."""
    s = np.maximum(s, float("1e-12"))
    X = np.column_stack([np.ones_like(x), x])
    W = 1.0r / (s * s)
    C = np.linalg.inv(X.T @ (X * W[:, None]))
    b = C @ (X.T @ (W * y))
    chi2 = float((W * (y - X @ b) ** 2r).sum())
    dof = len(x) - 2r
    red = chi2 / dof if dof > 0 else float("nan")
    scale = math.sqrt(red) if (dof > 0 and red > 1) else 1.0r
    return float(b[0]), math.sqrt(C[0, 0]) * scale, float(b[1]), math.sqrt(C[1, 1]) * scale, red


def series(rawd, p, N_values):
    means, ses = [], []
    for N in N_values:
        v = np.array(rawd[N][p], dtype=float) / float(N)
        m = float(v.mean()) if len(v) > 0 else float("nan")
        se = float(v.std(ddof=1r) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
        means.append(m)
        ses.append(se)
    return np.array(means), np.array(ses)


def fit_models(N_values, means, ses):
    """Za svaki model u MODELS vraca (c, ec, b, eb, chi2dof) ili None ako nema dovoljno tacaka."""
    ok = np.isfinite(means) & np.isfinite(ses)
    Narr = np.array(N_values, dtype=float)[ok]
    res = {}
    for name, ex in MODELS:
        if ok.sum() < 3r:
            res[name] = None
        else:
            res[name] = wls(Narr ** (-ex), means[ok], ses[ok])
    return res


def pm(c, e):
    return f"{c:.4f}±{e:.4f}"


def trend_text(fit):
    """Znak i znacajnost nagiba u modelu 1/sqrt(N): a>0 => c(N) pada sa N."""
    if fit is None:
        return "?"
    c, ec, b, eb, red = fit
    z = b / eb if eb > 0 else 0.0
    if abs(z) < 2:
        return "ravno"
    return "pada sa N" if b > 0 else "raste sa N"


def analyze_rgg(raw, kk, degrees, N_values):
    csv_rows = []
    summary = []
    for d in degrees:
        cnt = [len(raw[d][N][PRIMES[0]]) for N in N_values]
        out = []
        out.append("=" * 100)
        out.append(f"d = {d}   (r = sqrt(d/(pi N)), torus)   grafova po N: {cnt}")
        out.append("=" * 100)
        out.append("p-rk/N,  p-rk = N - rank_p(L) - k   (definicija A)")
        out.append("p".rjust(3) + "".join((f"N={N}").rjust(9) for N in N_values))
        se_rows = []
        fit_lines = []
        for p in PRIMES:
            means, ses = series(raw[d], p, N_values)
            out.append(str(p).rjust(3) + "".join(fmt4(m).rjust(9) for m in means))
            se_rows.append(str(p).rjust(3) + "".join(fmt4(e).rjust(9) for e in ses))
            for N, m, e in zip(N_values, means, ses):
                csv_rows.append(["A", d, N, p, len(raw[d][N][p]), m, e])
            fits = fit_models(N_values, means, ses)
            f1, f2 = fits["1/N"], fits["1/sqrt(N)"]
            if f1 is not None and f2 is not None:
                syst = abs(f1[0] - f2[0])
                fit_lines.append(f"{p:>3}  1/N: {pm(f1[0], f1[1])} (chi2/dof {f1[4]:.1f})   "
                                 f"1/sqrt(N): {pm(f2[0], f2[1])} (chi2/dof {f2[4]:.1f})   "
                                 f"syst {syst:.4f}   {trend_text(f2)}")
                summary.append((d, p, f1[0], f1[1], f2[0], f2[1], f1[4], f2[4]))
        out.append("standardna greska (SEM):")
        out += se_rows
        out.append("")
        out.append("Ekstrapolacija c_inf (tezinski fit):")
        out += fit_lines
        out.append("")
        out.append("k/N (komponente / N): " + " ".join(
            fmt4(np.mean(kk[d][N]) / float(N)) if len(kk[d][N]) else "nan" for N in N_values))
        text = "\n".join(out)
        print("\n" + text)
        with open(f"{OUT}_d" + str(d).replace(".", "_") + ".txt", "w") as f:
            f.write(text + "\n")
    return csv_rows, summary


def save_csv(rows, filename):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["definition", "d", "N", "p", "n_graphs", "mean_prk_over_N", "stderr"])
        for row in rows:
            w.writerow(row)
    print(f"\n[csv] {filename}")


def save_summary(summary, degrees, filename):
    S = []
    S.append("SAZETAK: ekstrapolirano c_p(d) = lim p-rk/N  (c_inf +- greska fita)")
    S.append("Dva modela: c_inf + a/N  i  c_inf + a/sqrt(N); 'syst' = |razlika modela|. "
             "Prijavi interval, ne tacan broj.")
    for p in PRIMES:
        S.append("")
        S.append(f"p = {p}")
        S.append("d".rjust(5) + "1/N".rjust(18) + "1/sqrt(N)".rjust(18) + "syst".rjust(9))
        for d in degrees:
            for row in summary:
                if row[0] == d and row[1] == p:
                    _, _, c1, e1, c2, e2, r1, r2 = row
                    S.append(f"{d:>5g}" + pm(c1, e1).rjust(18) + pm(c2, e2).rjust(18)
                             + f"{abs(c1 - c2):9.4f}")
    text = "\n".join(S)
    print("\n" + text)
    with open(filename, "w") as f:
        f.write(text + "\n")


def report_total_density(raw, degrees, N_values, filename):
    """S(N) = sum_p p-rk_p / N (zbir po svim prostim iz PRIMES), po grafu."""
    out = [f"UKUPNA GUSTINA: S(N) = sum_(p in {PRIMES}) p-rk/N",
           "d".rjust(5) + "".join((f"N={N}").rjust(9) for N in N_values) + "   S_inf(1/N)   S_inf(1/sqrt N)"]
    for d in degrees:
        means, ses = [], []
        for N in N_values:
            n_g = len(raw[d][N][PRIMES[0]])
            if n_g < 2r:
                means.append(float("nan")); ses.append(float("nan")); continue
            S = sum(np.array(raw[d][N][p], dtype=float) for p in PRIMES) / float(N)
            means.append(float(S.mean())); ses.append(float(S.std(ddof=1r) / math.sqrt(n_g)))
        means = np.array(means); ses = np.array(ses)
        fits = fit_models(N_values, means, ses)
        f1, f2 = fits["1/N"], fits["1/sqrt(N)"]
        tail = ""
        if f1 is not None and f2 is not None:
            tail = f"   {pm(f1[0], f1[1])}   {pm(f2[0], f2[1])}"
        out.append(f"{d:>5g}" + "".join(fmt4(m).rjust(9) for m in means) + tail)
    text = "\n".join(out)
    print("\n" + text)
    with open(filename, "w") as f:
        f.write(text + "\n")


def report_timing(tim, num_cores, wall):
    print("\nVREME: prosek po grafu (s) i ukupno CPU (s) po N")
    print("N".rjust(6) + "grafova".rjust(10) + "s/graf".rjust(10) + "CPU_s".rjust(10))
    cpu = float(0)
    for N in sorted(tim):
        v = tim[N]
        if not v:
            continue
        s = float(sum(v))
        cpu += s
        print(str(N).rjust(6) + str(len(v)).rjust(10) + ("%.3f" % (s / len(v))).rjust(10)
              + ("%.0f" % s).rjust(10))
    print(f"ukupno CPU {cpu:.0f}s na {num_cores} jezgara, zid {wall:.0f}s")


# Kontrola: slucajni d-regularni grafovi (Meszaros / Wood / Clancy-Leake-Payne)

def theory_P(p, r, K=200r):
    """Limes P(p-rk = r) = p^(-r(r+1)/2) prod_{i>r}(1-p^-i) / prod_{i>=1}(1-p^(-2i))."""
    p = float(p)
    r = int(r)
    num = p ** (-(r * (r + 1r)) / 2.0r)
    for i in range(r + 1r, K):
        num *= (1.0r - p ** (-float(i)))
    den = 1.0r
    for i in range(1r, K):
        den *= (1.0r - p ** (-2.0r * i))
    return num / den


def report_control(raw, kk, degrees, N, filename):
    out = [f"KONTROLA: slucajni d-regularni multigrafovi (unija d uparivanja), N={N}",
           "Ocekivano: (1) p-rk/N -> 0; (2) za NEPARNO d raspodela P(r) = formula (Wood/Meszaros Thm 1.2);",
           "           za PARNO d i p=2 formula ne vazi (2-rang je neparan), pa se p=2 preskace.", ""]
    for d in degrees:
        R = len(raw[d][N][PRIMES[0]])
        pc = np.mean([1.0r if k == 1r else 0.0r for k in kk[d][N]]) if R else float("nan")
        out.append(f"d={d}  grafova: {R}   udeo povezanih: {pc:.4f}")
        for p in PRIMES:
            if d % 2r == 0r and p == 2r:
                out.append(f"  p={p}: preskoceno (d parno)")
                continue
            v = np.array(raw[d][N][p], dtype=int)
            cnt = np.bincount(np.minimum(v, 5r), minlength=6r)[:6r]
            emp = cnt / float(R)
            se = np.sqrt(emp * (1.0r - emp) / float(R))
            line = f"  p={p}: mean p-rk = {v.mean():.3f}  (p-rk/N = {v.mean() / N:.5f})  "
            out.append(line)
            for r in range(4r):
                th = theory_P(p, r)
                z = (emp[r] - th) / se[r] if se[r] > 0 else float("nan")
                out.append(f"       r={r}: empirijski {emp[r]:.4f} +- {se[r]:.4f}   teorija {th:.4f}   z = {z:+.1f}")
        out.append("")
    text = "\n".join(out)
    print("\n" + text)
    with open(filename, "w") as f:
        f.write(text + "\n")


# Provera ispravnosti

def laplacian_entries(N, pairs):
    """Pun (tezinski) Laplasijan kao dict {(i,j): int}."""
    ent = {}
    deg = np.bincount(pairs.ravel(), minlength=N)
    for a, b in pairs.tolist():
        ent[(a, b)] = ent.get((a, b), 0r) - 1r
        ent[(b, a)] = ent.get((b, a), 0r) - 1r
    for i in range(N):
        if deg[i]:
            ent[(i, i)] = int(deg[i])
    return ent


def selftest():
    print("SELFTEST: brzi put vs naivni rank celog L (N-rank-k) i (N=60) vs Smith forma\n")
    rng = np.random.default_rng(int(1))
    bad = 0

    def check(label, N, pairs, use_snf):
        fast, k, ncore = p_ranks_from_pairs(N, pairs, PRIMES)
        G = Graph(N, multiedges=True)
        G.add_edges([(a, b) for a, b in pairs.tolist()])
        k_ref = int(G.connected_components_number())
        ent = laplacian_entries(N, pairs)
        naive = {}
        for p in PRIMES:
            Mp = dense_modp(N, {kk_: vv % p for kk_, vv in ent.items()}, p)
            naive[p] = int(N - int(Mp.rank()) - k_ref)
        snf = None
        if use_snf:
            divs = matrix(ZZ, N, N, ent, sparse=False).elementary_divisors()
            snf = {p: int(sum(1 for x in divs if x != 0 and int(x) % p == 0)) for p in PRIMES}
        ok = (k == k_ref) and all(fast[p] == naive[p] for p in PRIMES)
        if snf is not None:
            ok = ok and all(fast[p] == snf[p] for p in PRIMES)
        print(f"  {label:<14} k={k:<3} core={ncore:<4} fast={[fast[p] for p in PRIMES]} "
              f"naive={[naive[p] for p in PRIMES]}"
              + (f" snf={[snf[p] for p in PRIMES]}" if snf is not None else "")
              + ("   OK" if ok else "   *** NE POKLAPA SE ***"))
        return 0 if ok else 1

    for N, use_snf in ((60, True), (200, False)):
        for d in DEGREES:
            for rep in range(2):
                pairs = rgg_pairs(N, radius_for_degree(N, d), rng)
                bad += check(f"rgg N={N} d={d:g}", N, pairs, use_snf)
    for d in (3, 4, 5):
        for rep in range(3):
            pairs = regular_pairs(60, d, rng)
            bad += check(f"reg N=60 d={d}", 60, pairs, True)
    print("\nSELFTEST:", "SVE OK" if bad == 0 else f"{bad} neslaganja -- NE KORISTITI rezultate")


# Pokretanje

def run_control_mode(argv):
    R = int(argv[2]) if len(argv) > 2 else 1000
    N = int(argv[3]) if len(argv) > 3 else 1000
    cores = int(argv[4]) if len(argv) > 4 else (os.cpu_count() or 1)
    budget = float(argv[5]) if len(argv) > 5 else TIME_BUDGET
    degrees = [3, 4, 5]
    t0 = time.time()
    raw, kk, tim = run_experiment("reg", degrees, [N], {N: R}, cores, budget)
    report_control(raw, kk, degrees, N, f"{OUT}_control.txt")
    report_timing(tim, cores, time.time() - t0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        selftest()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "control":
        run_control_mode(sys.argv)
        sys.exit(0)

    factor = float(sys.argv[1]) if len(sys.argv) > 1 else float(1)
    num_cores = int(sys.argv[2]) if len(sys.argv) > 2 else (os.cpu_count() or 1)
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else TIME_BUDGET
    num_graphs = {N: max(2, int(round(float(NUM_GRAPHS_PER_N[N]) * factor))) for N in N_VALUES}

    t_start = time.time()
    print("=== EKSPERIMENT: p-RANK GUSTINA (RGG, torus, fiksno d), definicija A ===\n")
    raw, kk, tim = run_experiment("rgg", DEGREES, N_VALUES, num_graphs, num_cores, budget)
    csv_rows, summary = analyze_rgg(raw, kk, DEGREES, N_VALUES)
    save_csv(csv_rows, f"{OUT}_all.csv")
    save_summary(summary, DEGREES, f"{OUT}_summary.txt")
    report_total_density(raw, DEGREES, N_VALUES, f"{OUT}_total.txt")
    report_timing(tim, num_cores, time.time() - t_start)
    print(f"\nUkupno vreme: {time.time() - t_start:.0f} s")
    print(f"Grafici: python3 plot_prank.py {OUT}_all.csv --out figs")
