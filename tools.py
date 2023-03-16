def integrate(model, init, tmax, cg=None):
    """
    Iterate discrete time model

    Args
    ====
    model (func) function that describes the discrete time model with input as follow (n, t, pars)
    init (list) initial values
    tmax (int) maximum time
    cg (Config) parameters

    Return
    ======
    list of the time and population values
    """
    population = [init]
    t = 0
    t_series = [t]
    while t < tmax:
        pop_t = model(population[-1], cg) if cg is not None else model(population[-1])
        population.append(pop_t)
        t += 1
        t_series.append(t)
    return (t_series, population)
