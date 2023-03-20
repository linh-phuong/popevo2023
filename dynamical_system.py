def exp_discrete(n, R):
    return n + R * n


def exp_continuous(t, n, r):
    return r * n


def structural_population(n, leslie_matrix):
    return leslie_matrix @ n


def replicator_niche_overlap(t, n, pars):
    r = pars[:-1]
    alpha = pars[-1]
    n_total = n[-1]
    freq = n[:-1]
    gr = r - alpha * n_total
    r_bar = sum(gr * freq)
    return list((gr - r_bar) * freq) + [r_bar * n_total]


def replicator(t, n, pars):
    nb_sp = len(n) - 1
    freq = n[0:nb_sp]
    ntotal = n[-1]
    r = pars[0:nb_sp]
    A = pars[nb_sp:].reshape(nb_sp, nb_sp)
    gr = r - A@freq
    gr_bar = sum(gr * freq)
    return list((gr - gr_bar) * freq) + [gr_bar * ntotal]
