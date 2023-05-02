import streamlit as st
from dynamical_system import (
    invasion_fitness,
    invasion_fitness2,
    invasion_fitness3,
    pop_dynamics2,
)
import numpy as np
from tools import plot_3D_invfitness, plot_invasionfitness, make_interactive_video, plot_PIP
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Adaptive dynamics")

st.header("Example 1: When there is no cost on reproduction")
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

st.subheader("Invasion process video")

st.plotly_chart(
    make_interactive_video(0.01, zlist[-1], 50, zlist, invasion_fitness, alpha, [-2, 2])
)


st.subheader("Interactive step by step invasion and replacement")

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

st.header("Example 2: When there is cost in reproduction")

st.write(
    r"""
Now we include some cost in having a high intrinsical growth rate.

The ecological dynamics of the resident is now

$\frac{dn_r}{dt} = n_r(z_r - z_r^\beta - \alpha n_r)$

Do you see that now if the growth rate $z_r$ increases then there is an additional cost $z_r^\beta$.

The value of $\beta$ affect the shape of the cost
"""
)
zlist2 = np.linspace(0, 1, 100)

col_par1, col_par2 = st.columns(2, gap="large")
with col_par1:
    beta = st.slider(r"Value of $\beta$", 0.1, 2.0, value=1.2, step=0.01)
with col_par2:
    z_val2 = st.slider("Trait value", 0.0, 1.0, value=0.1, step=0.01)

z_star2 = (1 / beta) ** (1 / (beta - 1))

ndsol2 = solve_ivp(
    pop_dynamics2,
    (0, 550),
    [np.random.uniform(0, 0.05)],
    t_eval=np.linspace(0, 550, 200),
    args=((alpha, beta, z_val2),),
)

col5, col6 = st.columns(2, gap="large")
with col5:
    if ndsol2.y[0, -1] > 0:
        st.write(
            r"The population density reaches $\frac{z_r - z_r^\beta}{\alpha}$ = ", ndsol2.y[0, -1]
        )
    else:
        st.write("The population density reaches", 0)
    st.plotly_chart(
        go.Figure(
            data=go.Scatter(x=ndsol2.t, y=ndsol2.y[0, :], mode="lines"),
            layout=go.Layout(
                xaxis_title="Time", yaxis_title="Population dynamics", yaxis=dict(range=(0, 0.3))
            ),
        ),
    )
with col6:
    st.plotly_chart(
        go.Figure(
            data=[
                go.Scatter(x=zlist2, y=zlist2, name="Intrinsic growth rate"),
                go.Scatter(x=zlist2, y=zlist2**beta, name="Cost on mortality"),
            ]
        )
    )

st.subheader("Invasion process video")
st.write(
    r"""
The invasion fitness is $r(z_m, z) = z_m - z_m^\beta - (z - z^\beta)$
"""
)
z_start2 = st.number_input("Enter the start value of z then click play", 1e-5, 1.0, 0.1, step=0.01)
if z_start2 - z_start2**beta < 0:
    st.markdown(":red[POPULATION GOES EXTINCT]")
else:
    if beta > 1:
        z_end = z_star2
    elif beta < 1:
        zlist2 = np.linspace(0, 2, 100)
        z_end = zlist2[-1]
    st.plotly_chart(
        make_interactive_video(
            z_start2, z_end, 20, zlist, invasion_fitness2, (alpha, beta), [-0.2, 0.2]
        )
    )

st.write(" ")
st.subheader("Interactive step by step invasion replacement process")
st.write(
    r"""Now try yourself with the step by step invasion replacement process to verify the video

    Notice that when $$\beta < 1$$, the population goes extint at the singular strategy
"""
)

col7, col8, col9 = st.columns(3, gap="large")

with col7:
    zm2 = st.slider("Mutant trait value", 0.0, 1.0, value=0.1, step=0.01)

    st.plotly_chart(
        plot_invasionfitness(zm2, zlist2, invasion_fitness2, (alpha, beta), [-0.2, 0.2])
    )

X, Y = np.meshgrid(zlist2, zlist2)
inv_fitness3D2 = invasion_fitness2(X, Y, pars=(alpha, beta))

range2 = (np.min(inv_fitness3D2) - np.mean(inv_fitness3D2) - 1e-5, np.max(inv_fitness3D2))
with col8:
    st.plotly_chart(plot_PIP(zlist2, invasion_fitness2, (alpha, beta)))
with col9:
    st.plotly_chart(plot_3D_invfitness(zlist2, inv_fitness3D2, zm2, range2))


st.header("Example 3: Symetric competition")

st.write(
    r"""
Ssymetric competition results in evolutionary branching

Now the intrinsic growth rate is no longer a linear function of the trait z. We have

$\frac{dn_r}{dt} = (\rho(z) - \alpha(\delta_z) n_r) n_r$

where $\rho(z) = e^{-(z - z_0)^2}$ is a Gaussian function.

and $\alpha(\delta_z)$ = e^{- \frac{(\delta_z)^2}{\sigma}}.

In the resident population $\delta_z = z - z = 0$, hence, $\alpha = 1$. 
This means that competition among resident individuals is at the highest

"""
)
zlist3 = np.linspace(0, 10, 200)
z0 = 3
st.subheader("Reproduction and competition as a function of the trait")
col_par3, col_par4 = st.columns(2, gap="large")
with col_par3:
    st.plotly_chart(
        go.Figure(
            data=[
                go.Scatter(x=zlist3, y=np.exp(-((zlist3 - z0) ** 2) / 2)),
            ],
            layout=go.Layout(
                xaxis_title="Trait value (z)",
                yaxis_title="Intrinsic growth",
                xaxis=dict(range=(0, 8)),
            ),
        )
    )
with col_par4:
    k = st.slider("Competition width", 0.5, 4.0, 1.5, 0.1)
    dz = np.linspace(-10, 10.0, 100)
    st.plotly_chart(
        go.Figure(
            data=[
                go.Scatter(x=dz, y=np.exp(-(dz**2) / k)),
            ],
            layout=go.Layout(
                xaxis_title=r"Trait difference ($\delta_z$)", yaxis_title="Competition strength"
            ),
        )
    )
st.subheader("Invasion process video")
st.write(
    r"""
If a mutant arises, it has a different trait value than the resident, 
hence $\delta_z = z_m - z_r$, and the competition coefficient is

$\alpha = e^{- \frac{(z_m - z)^2}{\sigma}}$
"""
)
z_start3 = st.number_input("Enter mutant trait value and click play", 0.1, 10.0, 3.1, 0.5)
st.plotly_chart(
    make_interactive_video(z_start3, z0, 20, zlist3, invasion_fitness3, (z0, k), (-1, 1))
)
st.subheader("Interactive step by step invasion process")
st.write(
    """
    Some suggestions to play with the interaction:

    - Try to vary the competition width to see if the PIP change?

    - Does the nature of the singular strategy change when the competition width changes? Can you explain why?

    - Set the value of the competition width at 1.4, what happens to the evolutionary processes?
    Does it converges to the singular strategy? At the singular strategy, can other strategies invade?

"""
)
col10, col11, col12 = st.columns(3, gap="large")
with col10:
    zm3 = st.slider("Mutant trait value", 0.0, 10.0, 0.1)
    st.plotly_chart(plot_invasionfitness(zm3, zlist3, invasion_fitness3, (z0, k), (-1, 1)))
with col11:
    X, Y = np.meshgrid(zlist3, zlist3)
    inv_fitness3D3 = invasion_fitness3(X, Y, (z0, k))
    st.plotly_chart(plot_PIP(zlist3, invasion_fitness3, (z0, k)))
with col12:
    range3 = (np.min(inv_fitness3D3) - np.mean(inv_fitness3D3) - 1e-5, np.max(inv_fitness3D3))
    st.plotly_chart(plot_3D_invfitness(zlist3, inv_fitness3D3, zm3, range3))
