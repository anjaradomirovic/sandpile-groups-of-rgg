# erdosExp.sage
#
# Dva odvojena eksperimenta nad slucajnim Erdos-Renyi grafovima G(N, p=0.1):
#
#   1) EKSPERIMENT 2-RANK:
#      Za svaki graf racuna se samo 2-rank(G) = (N-1) - rank(L mod 2),
#      gde je L reciklirani (reduced) Laplasijan. Prati se srednja
#      vrednost 2-rank-a preko 100 grafova, za svako N.
#
#   2) EKSPERIMENT CIKLICNOSTI:
#      Za svaki graf se proverava da li je kriticna grupa ciklicna, tj.
#      da li je p-rank(G) <= 1 za SVE proste p <= N. Prati se udeo
#      (cyclic_fraction) grafova koji su ciklicni, za svako N.
#
# Oba eksperimenta koriste iste pomocne funkcije (reduced_laplacian,
# p_rank_value), ali se generisanje grafova i racunanje statistike
# rade potpuno nezavisno jedno od drugog - svaki eksperiment generise
# svoj sopstveni skup slucajnih grafova.
#
# Pokretanje (u terminalu, Ubuntu):
#   sage erdosExp.sage
#
# Ili u VS Code: otvoriti fajl, pa u integrisanom terminalu pokrenuti
# istu komandu (potrebno je da je SageMath instaliran i dostupan u PATH-u).


import csv

# Zajednicke funkcije

def reduced_laplacian(G):
    L = G.laplacian_matrix()
    n = L.nrows()
    return L.delete_rows([n - 1]).delete_columns([n - 1])


def p_rank_value(Lr, p, N):
    """p-rank(G) = (N-1) - rank(L mod p), racunato nad konacnim poljem GF(p)."""
    Lp = Lr.change_ring(GF(p))
    r = Lp.rank()
    return (N - 1) - r


def random_connected_graph(N, density):
    G = graphs.RandomGNP(N, density)
    while not G.is_connected():
        G = graphs.RandomGNP(N, density)
    return G


# EKSPERIMENT 1: 2-RANK

def two_rank_of_graph(G, N):
    """Racuna samo 2-rank(G) = (N-1) - rank(L mod 2)."""
    Lr = reduced_laplacian(G)
    return p_rank_value(Lr, 2, N)


def run_two_rank_experiment(N_values, num_graphs=1000, density=0.1, verbose=True):
    results = {}

    total_two_rank = 0
    total_graph_count = 0

    for N in N_values:

        two_ranks = []

        for i in range(num_graphs):
            G = random_connected_graph(N, density)

            two_rank = two_rank_of_graph(G, N)

            two_ranks.append(two_rank)
            total_two_rank += two_rank
            total_graph_count += 1

            if verbose and (i + 1) % 20 == 0:
                print(f"  [2-rank] N={N}: obradjeno {i + 1}/{num_graphs} grafova")

        mean_two_rank = sum(two_ranks) / len(two_ranks)

        results[N] = {
            "mean_two_rank": mean_two_rank,
            "two_ranks": two_ranks,
        }

        print(f"[2-rank] N={N:>4}  |  mean_2_rank = {float(mean_two_rank):.3f}")

    overall_mean_two_rank = total_two_rank / total_graph_count

    print("\n[2-rank] UKUPNO:")
    print(f"Broj grafova = {total_graph_count}")
    print(f"Mean 2-rank = {float(overall_mean_two_rank):.3f}")

    return results


def save_two_rank_csv(results, filename="erdosExp_two_rank.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "mean_two_rank"])
        for N, data in results.items():
            writer.writerow([N, float(data["mean_two_rank"])])
    print(f"\n[2-rank] Rezultati sacuvani u: {filename}")


# EKSPERIMENT 2: CIKLICNOST 

def is_cyclic_critical_group(G, N, primes_list):
    """Vraca True ako je p-rank(G) <= 1 za SVE proste p <= N.
    Cim se nadje p-rank > 1, prekida se dalje racunanje (early exit)."""
    Lr = reduced_laplacian(G)

    for p in primes_list:
        pr = p_rank_value(Lr, p, N)
        if pr > 1:
            return False  # dalje proveravanje nepotrebno - graf je vec neciklican

    return True


def run_cyclicity_experiment(N_values, num_graphs=1000, density=0.1, verbose=True):
    results = {}

    total_cyclic_count = 0
    total_graph_count = 0

    for N in N_values:

        # svi prosti brojevi p, 0 <= p <= N  (prime_range(N+1) vraca p < N+1)
        primes_list = prime_range(N + 1)

        cyclic_count = 0

        for i in range(num_graphs):
            G = random_connected_graph(N, density)

            if is_cyclic_critical_group(G, N, primes_list):
                cyclic_count += 1
                total_cyclic_count += 1

            total_graph_count += 1

            if verbose and (i + 1) % 20 == 0:
                print(f"  [cyclic] N={N}: obradjeno {i + 1}/{num_graphs} grafova")

        cyclic_fraction = cyclic_count / num_graphs

        results[N] = {
            "cyclic_fraction": cyclic_fraction,
        }

        print(f"[cyclic] N={N:>4}  |  cyclic_fraction = {float(cyclic_fraction):.3f}")

    overall_cyclic_fraction = total_cyclic_count / total_graph_count

    print("\n[cyclic] UKUPNO:")
    print(f"Broj grafova = {total_graph_count}")
    print(f"Cyclic fraction = {float(overall_cyclic_fraction):.3f}")

    return results


def save_cyclicity_csv(results, filename="erdosExp_cyclicity.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "cyclic_fraction"])
        for N, data in results.items():
            writer.writerow([N, float(data["cyclic_fraction"])])
    print(f"\n[cyclic] Rezultati sacuvani u: {filename}")


# Pokretanje

if __name__ == "__main__":
    N_values = [100, 150, 200, 250, 300]
    num_graphs = 1000
    density = 0.1

    print("=== EKSPERIMENT 1: 2-RANK ===\n")
    two_rank_results = run_two_rank_experiment(N_values, num_graphs=num_graphs, density=density)
    save_two_rank_csv(two_rank_results)

    print("\n\n=== EKSPERIMENT 2: CIKLICNOST KRITICNE GRUPE ===\n")
    cyclicity_results = run_cyclicity_experiment(N_values, num_graphs=num_graphs, density=density)
    save_cyclicity_csv(cyclicity_results)
    