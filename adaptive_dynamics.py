import streamlit as st
from dynamical_system import invasion_fitness, invasion_fitness2, invasion_fitness3, pop_dynamics2
import numpy as np
from tools import plot_3D_invfitness, plot_invasionfitness, make_interactive_video, plot_PIP
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Adaptive dynamics")

st.subheader("Example 1: When there is no cost on reproduction")
st.write(
    r"""
The ecological dynamics of the resident is

$\frac{dn_r}{dt} = n_r(z_r - \alpha n_r)$

where $z_r$ is the intrinsic growth rate of the resident, 
$\alpha$ is the competition coefficient among resident individual

The resident reaches equilibrium $n^* = \frac{z_r}{\alpha}$ before a mutant arises.
New mutant with a different intrinsic growth rate $z_m$ arises has dynamics as followed

$\frac{dn_m}{dt} = n_m(z_m - \alpha (n_r + n_m))$

The mutant growth rate is $r(z_m, z) = z_m - \alpha n^*$. 
This is also the invasion fitness of the mutant.
The invasion fitness depends on the mutant's trait value $z_m$, 
and the resident's trait value $z_r$ through the resident density at equilibrium $n^*$

The video shows the process of mutant invasion and replacement
"""
)

zlist = np.linspace(0, 2, 100)
alpha = 0.4

X, Y = np.meshgrid(zlist, zlist)
inv_fitness3D = invasion_fitness(X, Y, pars=alpha)

st.write("Invasion process video")

st.plotly_chart(
    make_interactive_video(0.01, zlist[-1], 50, zlist, invasion_fitness, alpha, [-2, 2])
)


st.write(
    """
Now you can try varying the mutant trait value to see how the fitness landscape change.

Some suggestions for you to think:

- In which case mutants with smaller growth rate value invade?

- Do you see the relateness among the three graphs? 

Hint: Rotate the 3D graph to match with the axes of the first two graph to see if:

The first graph is the vertical slice (gray surface) in the last graph

PIP is the projection of the fitness surface in the last graph, 

"""
)

zm = st.slider("Mutant trait value", 0.0, 2.0, value=0.2, step=0.01)

col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(plot_invasionfitness(zm, zlist, invasion_fitness, alpha, [-2, 2]))
with col2:
    st.plotly_chart(plot_PIP(zlist, invasion_fitness, alpha))
with col3:
    st.plotly_chart(plot_3D_invfitness(zlist, inv_fitness3D, zm, (-4, 4)))

st.header("When there is cost in reproduction")

st.write(
    r"""
Now we include some cost in having a high intrinsical growth rate.

The ecological dynamics of the resident is now

$\frac{dn_r}{dt} = n_r(z_r - z_r^\beta - \alpha n_r)$

Do you see that now if the growth rate $z_r$ increases then there is an additional cost $z_r^\beta$.

The value of $\beta$ affect the shape of the cost
"""
)
zlist = np.linspace(0, 1, 100)

col_par1, col_par2 = st.columns(2, gap="large")
with col_par1:
    beta = st.slider(r"Value of $\beta$", 0.1, 2.0, value=1.2, step=0.01)
with col_par2:
    z_val = st.slider("Trait value", 0.0, 1.0, value=0.1, step=0.01)

z_star = (1 / beta) ** (1 / (beta - 1))

ndsol = solve_ivp(
    pop_dynamics2,
    (0, 550),
    [np.random.uniform(0, 0.05)],
    t_eval=np.linspace(0, 550, 200),
    args=((alpha, beta, z_val),),
)

col5, col6 = st.columns(2, gap="large")
with col5:
    if ndsol.y[0, -1] > 0:
        st.write("The population density reaches", ndsol.y[0, -1])
    else:
        st.write("The population density reaches", 0)
    st.plotly_chart(
        go.Figure(
            data=go.Scatter(x=ndsol.t, y=ndsol.y[0, :], mode="lines"),
            layout=go.Layout(
                xaxis_title="Time", yaxis_title="Population dynamics", yaxis=dict(range=(0, 0.3))
            ),
        ),
    )
with col6:
    st.plotly_chart(
        go.Figure(
            data=[
                go.Scatter(x=zlist, y=zlist, name="Intrinsic growth rate"),
                go.Scatter(x=zlist, y=zlist**beta, name="Cost on mortality"),
            ]
        )
    )

col7, col8 = st.columns(2, gap="large")
with col7:
    st.write("Invasion process video")
    z_start = st.number_input(
        "Enter the start value of z then click play", 1e-5, 1.0, 0.1, step=0.01
    )
    st.plotly_chart(
        make_interactive_video(
            z_start, z_star, 20, zlist, invasion_fitness2, (alpha, beta), [-0.2, 0.2]
        )
    )
with col8:
    zm2 = st.slider("Mutant trait value", 0.0, 1.0, value=0.1, step=0.01)

    st.plotly_chart(plot_invasionfitness(zm2, zlist, invasion_fitness2, (alpha, beta), [-0.2, 0.2]))

X, Y = np.meshgrid(zlist, zlist)
inv_fitness3D2 = invasion_fitness2(X, Y, pars=(alpha, beta))

col9, col10 = st.columns(2, gap="large")
range = (np.min(inv_fitness3D2) - np.mean(inv_fitness3D2) - 1e-5, np.max(inv_fitness3D2))
with col9:
    st.plotly_chart(plot_3D_invfitness(zlist, inv_fitness3D2, zm, range))
with col10:
    st.plotly_chart(plot_PIP(zlist, invasion_fitness2, (alpha, beta)))


st.header("Assymetric competition")
zlist = np.linspace(-3, 3, 100)
inv_fitness3D3 = invasion_fitness3(zlist, zlist, (0, 1.4))
zm3 = st.slider("Mutant trait value", -3.0, 3.0, 0.1)
st.plotly_chart(plot_invasionfitness(zm3, zlist, invasion_fitness3, (0, 1.4), (-1, 1)))
st.plotly_chart(plot_PIP(zlist, invasion_fitness3, (0, 1.4)))
