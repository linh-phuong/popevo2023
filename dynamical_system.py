def exp_discrete(n, R):
    return n + R * n


def exp_continuous(t, n, r):
    return r * n


def structural_population(n, leslie_matrix):
    return leslie_matrix @ n
