# entropyExp.sage
#
# Paralelni eksperiment: tree entropy za Erdos-Renyi G(N, p) i za RGG(N, r).
#
#   tree entropy = (1/N) * log |Pic(G)|
#
# |Pic(G)| = broj razapinjucih stabala grafa G = det(reduced Laplasijan)
# (Kirchhoff-ova matrix-tree teorema). Determinanta se racuna TACNO nad ZZ,
# a tek zatim se uzima logaritam (broj je ogroman, ali RR to podnosi).
#
# Za svako N se u istoj petlji, u svakoj iteraciji, generise po jedan
# povezan ER graf i jedan povezan RGG graf, pa se rezultati porede.
#
#   ER:  G(N, p=0.1)
#   RGG: N tacaka u [0,1]^2, grana ako je rastojanje < r,
#        r = c * r_c(N),  r_c(N) = sqrt(log(N) / (pi*N)),  c = 3
#
# Pokretanje:
#   sage entropyExp.sage

import csv
import math
import random


# Generisanje grafova

def random_connected_er(N, density):
    G = graphs.RandomGNP(N, density)
    while not G.is_connected():
        G = graphs.RandomGNP(N, density)
    return G


def connectivity_threshold(N):
    return math.sqrt(math.log(N) / (math.pi * N))


def random_geometric_graph(N, r):
    pts = [(random.random(), random.random()) for _ in range(N)]
    G = Graph()
    G.add_vertices(range(N))
    r2 = r * r
    for i in range(N):
        xi, yi = pts[i]
        for j in range(i + 1, N):
            dx = xi - pts[j][0]
            dy = yi - pts[j][1]
            if dx * dx + dy * dy < r2:
                G.add_edge(i, j)
    return G


def random_connected_rgg(N, r, max_attempts=200):
    for _ in range(max_attempts):
        G = random_geometric_graph(N, r)
        if G.is_connected():
            return G
    raise RuntimeError(
        f"N={N}, r={r:.4f}: nije dobijen povezan RGG posle {max_attempts} "
        f"pokusaja (r_c={connectivity_threshold(N):.4f}) - povecaj radius_multiplier."
    )


# Tree entropy

def reduced_laplacian(G):
    L = G.laplacian_matrix()
    n = L.nrows()
    return L.delete_rows([n - 1]).delete_columns([n - 1])


def tree_entropy(G):
    """(1/n) * log |Pic(G)|, |Pic(G)| = det(reduced Laplacian)."""
    n = G.order()
    Lr = reduced_laplacian(G).change_ring(ZZ)
    pic = Lr.det()                      # tacan broj razapinjucih stabala
    return float(RR(pic).log() / n)


def mean_and_std(xs):
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
    return m, math.sqrt(var)


# Eksperiment

def run_entropy_experiment(N_values, num_graphs=100, density=0.1,
                           radius_multiplier=3.0, verbose=True):
    results = {}

    for N in N_values:
        r = radius_multiplier * connectivity_threshold(N)

        er_ent, rgg_ent = [], []
        er_deg, rgg_deg = [], []

        for i in range(num_graphs):
            G_er = random_connected_er(N, density)
            G_rgg = random_connected_rgg(N, r)

            er_ent.append(tree_entropy(G_er))
            rgg_ent.append(tree_entropy(G_rgg))
            er_deg.append(2 * G_er.size() / G_er.order())
            rgg_deg.append(2 * G_rgg.size() / G_rgg.order())

            if verbose and (i + 1) % 20 == 0:
                print(f"  N={N}: obradjeno {i + 1}/{num_graphs} parova grafova")

        er_m, er_s = mean_and_std(er_ent)
        rgg_m, rgg_s = mean_and_std(rgg_ent)

        results[N] = {
            "r": r,
            "er_mean": er_m, "er_std": er_s,
            "rgg_mean": rgg_m, "rgg_std": rgg_s,
            "er_avg_deg": sum(er_deg) / num_graphs,
            "rgg_avg_deg": sum(rgg_deg) / num_graphs,
            # teorijska aproksimacija za ER: tau ~ N^(N-2) p^(N-1)  =>  ~ log(N p)
            "er_theory": math.log(N * density),
        }

        print(f"N={N:>4} | ER: {er_m:.4f} (sd {er_s:.4f}, deg {results[N]['er_avg_deg']:.1f})"
              f" | RGG: {rgg_m:.4f} (sd {rgg_s:.4f}, deg {results[N]['rgg_avg_deg']:.1f}, r={r:.4f})")

    return results


def print_comparison(results):
    print("\n=== POREDJENJE: tree entropy (1/N) log|Pic(G)| ===")
    print(f"{'N':>5} {'ER mean':>9} {'RGG mean':>9} {'ER-RGG':>9} "
          f"{'ER deg':>7} {'RGG deg':>8} {'log(Np)':>8}")
    for N, d in results.items():
        print(f"{N:>5} {d['er_mean']:>9.4f} {d['rgg_mean']:>9.4f} "
              f"{d['er_mean'] - d['rgg_mean']:>9.4f} "
              f"{d['er_avg_deg']:>7.1f} {d['rgg_avg_deg']:>8.1f} {d['er_theory']:>8.4f}")


def save_csv(results, filename="entropyExp.csv"):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "r", "er_mean_entropy", "er_std", "er_avg_degree",
                    "rgg_mean_entropy", "rgg_std", "rgg_avg_degree", "er_theory_log_Np"])
        for N, d in results.items():
            w.writerow([N, d["r"], d["er_mean"], d["er_std"], d["er_avg_deg"],
                        d["rgg_mean"], d["rgg_std"], d["rgg_avg_deg"], d["er_theory"]])
    print(f"\nRezultati sacuvani u: {filename}")


# Pokretanje

if __name__ == "__main__":
    N_values = [100, 150, 200, 250, 300]
    num_graphs = 100          # po N: 100 ER + 100 RGG grafova
    density = 0.1             # ER: p
    radius_multiplier = 3.0   # RGG: r = 3 * r_c(N)

    print("=== EKSPERIMENT: TREE ENTROPY (ER vs RGG) ===\n")
    res = run_entropy_experiment(N_values, num_graphs=num_graphs,
                                 density=density, radius_multiplier=radius_multiplier)
    print_comparison(res)
    save_csv(res)
