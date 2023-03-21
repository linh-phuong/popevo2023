import streamlit as st
import numpy as np
from tools import integrate, make_plot
from dynamical_system import exp_discrete, exp_continuous
from scipy.integrate import odeint
import pandas as pd
import altair as alt

st.set_page_config(layout="wide")

st.title("Discrete vs continuous population")

st.header("Exponential growth population")

st.markdown(
    r"""
Discrete time model

$N_{t+1} = N_t + R N_t$

Continuous time model

$\frac{dN}{dt} = r N$


"""
)

st.write("")

init = 0.2  # initial population density
time_series = np.arange(0, 101, 1)

r = st.sidebar.number_input(
    "r", 1e-3, 0.2, value=0.05, step=1e-4, format="%.4f")
R = st.sidebar.number_input(
    "R", 1e-3, 0.2, value=0.05, step=1e-4, format="%.4f")
st.sidebar.write(
    r"The two models give similar results when $R = e^r - 1 = $", np.exp(r) - 1
)

# simulation for continuous time model
pop_continuous = odeint(exp_continuous, init,
                        time_series, args=(r,), tfirst=True)

# simulation for discrete time model
t_discrete, pop_discrete = integrate(exp_discrete, init, time_series[-1], R)

df = pd.DataFrame(
    dict(
        time=t_discrete, pop_discrete=pop_discrete, pop_continuous=pop_continuous[:, 0]
    )
)

st.write(
    "The following graphs show the intergration of discrete time model (blue cirles) and continuous time model (yellow line)"
)

st.write("")


col1, col2 = st.columns(2, gap="large")

with col1:
    st.altair_chart(make_plot(df, "linear", "Normal scale"))
with col2:
    st.altair_chart(make_plot(df, "log", "Log scale"))
