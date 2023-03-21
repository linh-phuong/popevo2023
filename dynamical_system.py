def exp_discrete(n, R):
    return n + R * n


def exp_continuous(t, n, r):
    return r * n


def structural_population(n, leslie_matrix):
    return leslie_matrix @ n


def replicator(t, n, pars):
    nb_sp = len(n) - 1
    freq = n[0:nb_sp]
    ntotal = n[-1]
    r = pars[0:nb_sp]
    competitions = pars[nb_sp:]
    if len(competitions) > 1:
        A = competitions.reshape(nb_sp, nb_sp)
        gr = r - A @ (freq * ntotal)
    else:
        gr = r - competitions * ntotal
    gr_bar = sum(gr * freq)
    return list((gr - gr_bar) * freq) + [gr_bar * ntotal]
