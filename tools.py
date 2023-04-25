import altair as alt
import pandas as pd
import plotly.graph_objects as go
import numpy as np


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
    l2 = alt.Chart(df).mark_line().encode(x="time", y="pop_continuous", color=alt.value("#ff7f0e"))
    return l1 + l2


def make_dataframe(raw_dat):
    df = pd.DataFrame(
        dict(
            time=raw_dat.t,
            type_1=raw_dat.y[0, :],
            type_2=raw_dat.y[1, :],
            type_3=raw_dat.y[2, :],
            total=raw_dat.y[3, :],
        )
    )
    return df


def plot_3D_invfitness(trait, fitness, resident, color="Greens"):
    X, Y = np.meshgrid(trait, trait)
    f_projection = (np.min(fitness) - 2) * np.ones(fitness.shape)
    axis = dict(
        showbackground=True,
        backgroundcolor="rgb(230, 230,230)",
        showgrid=False,
        zeroline=False,
        showline=False,
    )

    layout = go.Layout(
        autosize=False,
        width=700,
        height=600,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis, range=(-14, 14)),
            aspectratio=dict(x=1, y=1, z=1),
            xaxis_title="Resident trait",
            yaxis_title="Mutant trait",
            zaxis_title="Invasion fitness",
        ),
    )
    x_projection = resident * np.ones(fitness.shape)
    fitness_surface = go.Surface(x=X, y=Y, z=fitness, colorscale=color)
    PIP = go.Surface(
        x=X,
        y=Y,
        z=f_projection,
        surfacecolor=(fitness > 0),
        colorscale=color,
        showlegend=False,
        showscale=False,
    )
    slice = go.Surface(
        x=x_projection,
        y=Y,
        z=(fitness * 5),
        surfacecolor=x_projection,
        colorscale="Greys",
        opacity=0.5,
        showlegend=False,
        showscale=False,
    )
    fig = go.Figure(
        data=[
            fitness_surface,
            PIP,
            slice,
        ],
        layout=layout,
    )
    return fig
