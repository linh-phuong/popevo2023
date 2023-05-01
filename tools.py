import altair as alt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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


def plot_3D_invfitness(trait, fitness, resident, range, color="RdBu"):
    X, Y = np.meshgrid(trait, trait)
    f_projection = (np.min(fitness) - np.mean(fitness)) * np.ones(fitness.shape)
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
            zaxis=dict(axis, range=range),
            aspectratio=dict(x=1, y=1, z=1),
            xaxis_title="Resident trait (z_r)",
            yaxis_title="Mutant trait (z_m)",
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
        colorscale="Greens",
        showlegend=False,
        showscale=False,
    )
    slice = go.Surface(
        x=x_projection,
        y=Y,
        z=(fitness * 1e5),
        surfacecolor=x_projection,
        colorscale="Greys",
        opacity=0.7,
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


def plot_invasionfitness(zm, zlist, fitness_func, pars, range):
    inv_fitness = fitness_func(zm, zlist, pars)

    fig = px.line(
        x=zlist, y=inv_fitness, labels={"x": "Mutant trait value (z_m)", "y": "Invasion fitness"}
    )
    fig.add_vline(x=zm, line_dash="dashdot")
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        title="Interactive invasion process",
        xaxis=dict(range=[0, zlist[-1]], autorange=False),
        yaxis=dict(range=range, autorange=False),
        autosize=False,
        width=450,
        height=400,
    )
    return fig


def make_interactive_video(z_start, z_end, steps, zlist, fitness_func, pars, range):
    inv_vid = []
    for z_val in np.linspace(z_start, z_end, steps):
        inv_vid.append(fitness_func(z_val, zlist, pars))
    vid = go.Figure(
        data=[
            go.Line(x=zlist, y=fitness_func(z_start, zlist, pars), name="invasion fitness"),
            go.Line(
                x=zlist,
                y=np.zeros(len(zlist)),
                line=dict(color="black", width=1, dash="dash"),
                name="Invasion threshold",
            ),
            go.Scatter(
                x=[z_start] * 10,
                y=np.linspace(-2, 2, 10),
                mode="lines",
                line=dict(color="black", dash="dashdot"),
                name="Resident trait value",
            ),
        ],
        layout=go.Layout(
            autosize=False,
            width=650,
            height=500,
            xaxis=dict(range=[0, zlist[-1]], autorange=False),
            yaxis=dict(range=range, autorange=False),
            xaxis_title="Mutant trait value (z_m)",
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 500, "redraw": False},
                                    "fromcurrent": True,
                                    "transition": {"duration": 300, "easing": "quadratic-in-out"},
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                ),
            ],
        ),
        frames=[
            go.Frame(
                data=[
                    go.Line(x=zlist, y=i),
                    go.Line(
                        x=zlist,
                        y=np.zeros(len(zlist)),
                        line=dict(color="black", dash="dash"),
                    ),
                    go.Scatter(
                        x=[z_val] * 10,
                        y=np.linspace(-2, 2, 10),
                        line=dict(color="black", dash="dashdot"),
                        mode="lines",
                    ),
                ]
            )
            for i, z_val in zip(inv_vid, np.linspace(z_start, z_end, steps))
        ],
    )
    return vid


def plot_PIP(zlist, fitness_func, pars):
    X, Y = np.meshgrid(zlist, zlist)
    inv_fitness3D = fitness_func(X, Y, pars)
    fig = go.Figure(
        data=go.Contour(
            x=zlist,
            y=zlist,
            z=inv_fitness3D,
            colorscale="PRGn",
            showscale=False,
            contours=dict(
                start=-20,
                end=0,
                size=10,
            ),
        )
    )
    fig.update_layout(
        autosize=False,
        width=400,
        height=500,
        xaxis_title=r"Resident trait (z_r)",
        yaxis_title=r"Mutant trait (z_m)",
        title="Pairwise invasibility plot (PIP)",
    )
    return fig
