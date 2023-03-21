import altair as alt
import pandas as pd


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
        pop_t = model(
            population[-1], cg) if cg is not None else model(population[-1])
        population.append(pop_t)
        t += 1
        t_series.append(t)
    return (t_series, population)


def make_plot(df, scale, title):
    tt = alt.TitleParams(title, anchor="middle")
    l1 = (
        alt.Chart(df, title=tt)
        .mark_point()
        .encode(
            x="time",
            y=alt.Y(
                "pop_discrete",
                axis=alt.Axis(title="Population density"),
                scale=alt.Scale(type=scale),
            ),
            color=alt.value("#1f77b4"),
        )
    )
    l2 = (
        alt.Chart(df)
        .mark_line()
        .encode(x="time", y="pop_continuous", color=alt.value("#ff7f0e"))
    )
    return l1 + l2


def make_dataframe(raw_dat):
    df = pd.DataFrame(
        dict(
            time=raw_dat.t,
            type_1=raw_dat.y[0, :],
            type_2=raw_dat.y[1, :],
            type_3=raw_dat.y[2, :],
            total=raw_dat.y[3, :],
        ))
    return df
