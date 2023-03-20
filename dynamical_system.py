def exp_discrete(n, R):
    return n + R * n


def exp_continuous(t, n, r):
    return r * n


def structural_population(n, leslie_matrix):
    return leslie_matrix @ n


# def replicator_exp(t, n, r):
#     n_total = n[-1]
#     freq = n[:-1]
#     r_bar = sum(r * freq)
#     return list((r - r_bar) * freq) + [r_bar * n_total]


def replicator_niche_overlap(t, n, pars):
    r = pars[:-1]
    alpha = pars[-1]
    n_total = n[-1]
    freq = n[:-1]
    growth = r - alpha * n_total
    r_bar = sum(growth * freq)
    return list((growth - r_bar) * freq) + [r_bar * n_total]
