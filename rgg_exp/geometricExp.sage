#
# Konstrukcija RGG(N, r):
#   - N tacaka se nasumicno i uniformno rasporedi u jedinicni kvadrat
#     [0,1] x [0,1];
#   - dva cvora se povezuju granom ako je Euklidsko rastojanje medju
#     njima strogo manje od r.
#
# RGG se raspada na vise komponenti ako je r ispod praga povezanosti
#
#       r_c(N) ~ sqrt( log(N) / (pi * N) )
#
# (standardni prag povezanosti za RGG u jedinicnom kvadratu - stepen
# cvora u proseku raste kao N*pi*r^2, pa iz N*pi*r^2 ~ log N sledi
# gornja formula). Da bismo dobili (skoro uvek) povezan graf, biramo
#
#       r = c * r_c(N),   c > 1  (podrazumevano c = 1.2)
#
# tj. radijus malo iznad praga povezanosti - ovo je analogno tome kako
# je u erdosExp.sage birana gustina p=0.1 pa se ceka da graf ispadne
# povezan (random_connected_graph). Ako se za dati (N, r) povezan graf
# ne dobije posle max_attempts pokusaja, kao fallback se uzima najveca
# povezana komponenta (giant component) tog grafa - tada efektivan
# broj cvorova bude manji od nominalnog N, i sve p-rank/2-rank formule
# se racunaju u odnosu na STVARAN broj cvorova grafa (G.order()), a ne
# u odnosu na nominalno N, pa ostaju korektne i u tom slucaju.
#
# Dva odvojena eksperimenta, ISTA logika kao u erdosExp.sage:
#
#   1) EKSPERIMENT 2-RANK:
#      Za svaki RGG graf racuna se 2-rank(G) = (n-1) - rank(L mod 2),
#      n = G.order(), L = reciklirani (reduced) Laplasijan. Prati se
#      srednja vrednost 2-rank-a preko num_graphs grafova, za svako N.
#
#   2) EKSPERIMENT CIKLICNOSTI:
#      Za svaki RGG graf se proverava da li je kriticna grupa ciklicna,
#      tj. da li je p-rank(G) <= 1 za SVE proste p <= n (n = G.order()).
#      Prati se udeo (cyclic_fraction) grafova koji su ciklicni, za
#      svako N.
#
# Oba eksperimenta koriste iste pomocne funkcije (reduced_laplacian,
# p_rank_value), ali generisanje grafova i racunanje statistike rade
# se potpuno nezavisno jedno od drugog - svaki eksperiment generise
# svoj sopstveni skup slucajnih RGG grafova.

import csv
import math
import random

# Pomocne funkcije - zajednicke za oba eksperimenta

def reduced_laplacian(G):
    L = G.laplacian_matrix()
    n = L.nrows()
    return L.delete_rows([n - 1]).delete_columns([n - 1])


def p_rank_value(Lr, p):
    """p-rank(G) = (n-1) - rank(L mod p), racunato nad GF(p).
    n se izvlaci direktno iz dimenzije reduced Laplasijana
    (n = Lr.nrows() + 1), tako da funkcija radi korektno i kada
    je stvaran broj cvorova grafa manji od nominalnog N (npr. kada
    se koristi giant component)."""
    n = Lr.nrows() + 1
    Lp = Lr.change_ring(GF(p))
    r = Lp.rank()
    return (n - 1) - r


# Generisanje RGG grafova

def connectivity_threshold(N):
    """r_c(N) = sqrt( log(N) / (pi * N) ) - standardni prag
    povezanosti za RGG u jedinicnom kvadratu."""
    return math.sqrt(math.log(N) / (math.pi * N))


def random_geometric_graph(N, r):
    """Rasporedi N tacaka uniformno u [0,1] x [0,1] i poveži svaki
    par cija je Euklidska udaljenost strogo manja od r."""
    pts = [(random.random(), random.random()) for _ in range(N)]

    G = Graph()
    G.add_vertices(range(N))

    r2 = r * r
    for i in range(N):
        xi, yi = pts[i]
        for j in range(i + 1, N):
            xj, yj = pts[j]
            dx = xi - xj
            dy = yi - yj
            if dx * dx + dy * dy < r2:
                G.add_edge(i, j)

    G.set_pos({i: pts[i] for i in range(N)})
    return G


def random_connected_rgg(N, r, max_attempts=200, verbose=False):
    """Generise RGG(N, r) i ponavlja dok graf ne ispadne povezan. Ako se posle
    max_attempts pokusaja ne dobije povezan graf, r je preblizu ili
    ispod praga povezanosti za dato N - baca se greska (povecaj radius_multiplier)"""
    for _ in range(max_attempts):
        G = random_geometric_graph(N, r)
        if G.is_connected():
            return G

    raise RuntimeError(
        f"N={N}, r={r:.4f}: nije dobijen povezan graf posle "
        f"{max_attempts} pokusaja. Radijus je verovatno preblizu ili "
        f"ispod praga povezanosti r_c(N)={connectivity_threshold(N):.4f} "
        f"- povecaj radius_multiplier."
    )


# EKSPERIMENT 1: 2-RANK


def two_rank_of_graph(G):
    """Racuna samo 2-rank(G) = (n-1) - rank(L mod 2)."""
    Lr = reduced_laplacian(G)
    return p_rank_value(Lr, 2)


def run_two_rank_experiment(N_values, num_graphs=1000, radius_multiplier=1.2, verbose=True):
    results = {}

    total_two_rank = 0
    total_graph_count = 0

    for N in N_values:

        r = radius_multiplier * connectivity_threshold(N)

        two_ranks = []

        for i in range(num_graphs):
            G = random_connected_rgg(N, r, verbose=verbose)

            two_rank = two_rank_of_graph(G)

            two_ranks.append(two_rank)
            total_two_rank += two_rank
            total_graph_count += 1

            if verbose and (i + 1) % 20 == 0:
                print(f"  [2-rank] N={N}: obradjeno {i + 1}/{num_graphs} grafova")

        mean_two_rank = sum(two_ranks) / len(two_ranks)

        results[N] = {
            "r": r,
            "mean_two_rank": mean_two_rank,
            "two_ranks": two_ranks,
        }

        print(f"[2-rank] N={N:>4}  r={r:.4f}  |  mean_2_rank = {float(mean_two_rank):.3f}")

    overall_mean_two_rank = total_two_rank / total_graph_count

    print("\n[2-rank] UKUPNO:")
    print(f"Broj grafova = {total_graph_count}")
    print(f"Mean 2-rank = {float(overall_mean_two_rank):.3f}")

    return results


def save_two_rank_csv(results, filename="geometricExp_two_rank.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "r", "mean_two_rank"])
        for N, data in results.items():
            writer.writerow([N, float(data["r"]), float(data["mean_two_rank"])])
    print(f"\n[2-rank] Rezultati sacuvani u: {filename}")


# EKSPERIMENT 2: CIKLICNOST KRITICNE GRUPE

def is_cyclic_critical_group(G):
    """Vraca True ako je p-rank(G) <= 1 za SVE proste p <= n = G.order().
    Cim se nadje p-rank > 1, prekida se dalje racunanje (early exit)."""
    n = G.order()
    Lr = reduced_laplacian(G)
    primes_list = prime_range(n + 1)

    for p in primes_list:
        pr = p_rank_value(Lr, p)
        if pr > 1:
            return False  # dalje proveravanje nepotrebno - graf je vec neciklican

    return True


def run_cyclicity_experiment(N_values, num_graphs=1000, radius_multiplier=1.2, verbose=True):
    results = {}

    total_cyclic_count = 0
    total_graph_count = 0

    for N in N_values:

        r = radius_multiplier * connectivity_threshold(N)

        cyclic_count = 0

        for i in range(num_graphs):
            G = random_connected_rgg(N, r, verbose=verbose)

            if is_cyclic_critical_group(G):
                cyclic_count += 1
                total_cyclic_count += 1

            total_graph_count += 1

            if verbose and (i + 1) % 20 == 0:
                print(f"  [cyclic] N={N}: obradjeno {i + 1}/{num_graphs} grafova")

        cyclic_fraction = cyclic_count / num_graphs

        results[N] = {
            "r": r,
            "cyclic_fraction": cyclic_fraction,
        }

        print(f"[cyclic] N={N:>4}  r={r:.4f}  |  cyclic_fraction = {float(cyclic_fraction):.3f}")

    overall_cyclic_fraction = total_cyclic_count / total_graph_count

    print("\n[cyclic] UKUPNO:")
    print(f"Broj grafova = {total_graph_count}")
    print(f"Cyclic fraction = {float(overall_cyclic_fraction):.3f}")

    return results


def save_cyclicity_csv(results, filename="geometricExp_cyclicity.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "r", "cyclic_fraction"])
        for N, data in results.items():
            writer.writerow([N, float(data["r"]), float(data["cyclic_fraction"])])
    print(f"\n[cyclic] Rezultati sacuvani u: {filename}")


# Pokretanje

if __name__ == "__main__":
    # Iste N vrednosti kao u erdosExp.sage
    N_values = [100, 150, 200, 250, 300]
    num_graphs = 1000
    radius_multiplier = 3.0  # r = radius_multiplier * r_c(N), iznad praga povezanosti

    print("=== EKSPERIMENT 1: 2-RANK (RGG) ===\n")
    two_rank_results = run_two_rank_experiment(
        N_values, num_graphs=num_graphs, radius_multiplier=radius_multiplier
    )
    save_two_rank_csv(two_rank_results)

    print("\n\n=== EKSPERIMENT 2: CIKLICNOST KRITICNE GRUPE (RGG) ===\n")
    cyclicity_results = run_cyclicity_experiment(
        N_values, num_graphs=num_graphs, radius_multiplier=radius_multiplier
    )
    save_cyclicity_csv(cyclicity_results)
