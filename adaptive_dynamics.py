import streamlit as st
from dynamical_system import invasion_fitness, invasion_fitness2, invasion_fitness3
import numpy as np
from tools import plot_3D_invfitness, plot_invasionfitness, make_interactive_video, plot_PIP

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


st.plotly_chart(
    make_interactive_video(0.01, zlist[-1], 0.03, zlist, invasion_fitness, alpha, [-2, 2])
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

beta = st.slider(r"Value of $\beta$", 0.1, 2.0, value=1.2, step=0.2)
col5, col6 = st.columns(2, gap="large")
with col5:
    st.plotly_chart(
        make_interactive_video(
            0.1, zlist[-1], 0.01, zlist, invasion_fitness2, (alpha, beta), [-0.2, 0.2]
        )
    )
with col6:
    zm2 = st.slider("Mutant trait value  ", 0.0, 1.0, value=0.1, step=0.01)
    st.plotly_chart(plot_invasionfitness(zm2, zlist, invasion_fitness2, (alpha, beta), [-0.2, 0.2]))

X, Y = np.meshgrid(zlist, zlist)
inv_fitness3D2 = invasion_fitness2(X, Y, pars=(alpha, beta))

col7, col8 = st.columns(2, gap="large")
range = (np.min(inv_fitness3D2) - np.mean(inv_fitness3D2) - 1e-5, np.max(inv_fitness3D2))
with col7:
    st.plotly_chart(plot_3D_invfitness(zlist, inv_fitness3D2, zm, range))
with col8:
    st.plotly_chart(plot_PIP(zlist, invasion_fitness2, (alpha, beta)))


st.header("Assymetric competition")
zlist = np.linspace(-3, 3, 100)
inv_fitness3D3 = invasion_fitness3(zlist, zlist, (0, 2.4))
st.plotly_chart(plot_PIP(zlist, invasion_fitness3, (0, 2.4)))
