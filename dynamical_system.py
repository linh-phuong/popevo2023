import numpy as np


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


def invasion_fitness(z, zm, pars):
    n_res = z / pars
    return zm - pars * n_res


def invasion_fitness2(z, zm, pars):
    alpha, beta = pars
    n_res = (z - z**beta) / alpha
    return zm - zm**beta - alpha * n_res


def invasion_fitness3(z, zm, pars):
    z0, k = pars
    r = np.exp(-((z - z0) ** 2) / 2)
    a = 1 / 2
    n_res = r / a
    rm = np.exp(-((zm - z0) ** 2) / 2)
    am = 1 - 1 / (1 + np.exp(-k * (zm - z)))
    return rm - am * n_res
