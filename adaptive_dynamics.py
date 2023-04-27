import streamlit as st
from dynamical_system import invasion_fitness, invasion_fitness2
import numpy as np
from tools import plot_3D_invfitness, plot_invasionfitness, make_interactive_video, plot_PIP

st.set_page_config(layout="wide")
st.title("Adaptive dynamics")

st.subheader("When there is no cost on reproduction")


zlist = np.linspace(0, 2, 100)
alpha = 0.4

X, Y = np.meshgrid(zlist, zlist)
inv_fitness3D = invasion_fitness(X, Y, pars=alpha)


col1, col2 = st.columns(2, gap="large")
with col1:
    st.plotly_chart(make_interactive_video(
        0.01, zlist[-1], 0.03, zlist, invasion_fitness, alpha, [-2, 2]))


with col2:
    zm = st.slider("Mutant trait value", 0.0, 2.0, value=0.2, step=0.01)
    st.plotly_chart(plot_invasionfitness(
        zm, zlist, invasion_fitness, alpha, [-2, 2]))

col3, col4 = st.columns(2, gap="large")
with col3:
    st.plotly_chart(plot_3D_invfitness(zlist, inv_fitness3D, zm, (-4, 4)))
with col4:
    st.plotly_chart(plot_PIP(zlist, invasion_fitness, alpha))

st.header("When there is cost in reproduction")

zlist = np.linspace(0, 1, 100)

beta = st.slider(r"Value of $\beta$", 0.1, 2.0, value=1.2, step=0.2)
col5, col6 = st.columns(2, gap="large")
with col5:
    st.plotly_chart(
        make_interactive_video(
            0.1, zlist[-1], 0.01, zlist, invasion_fitness2, (alpha, beta), [-0.2, 0.2])
    )
with col6:
    zm2 = st.slider("Mutant trait value  ", 0.0, 1.0, value=0.1, step=0.01)
    st.plotly_chart(plot_invasionfitness(
        zm2, zlist, invasion_fitness2, (alpha, beta), [-0.2, 0.2]))

X, Y = np.meshgrid(zlist, zlist)
inv_fitness3D2 = invasion_fitness2(X, Y, pars=(alpha, beta))

col7, col8 = st.columns(2, gap="large")
with col7:
    st.plotly_chart(plot_3D_invfitness(zlist, inv_fitness3D2, zm, (-3, 3)))
with col8:
    st.plotly_chart(plot_PIP(zlist, invasion_fitness2, (alpha, beta)))
