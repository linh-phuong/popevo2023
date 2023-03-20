import streamlit as st
from scipy.integrate import solve_ivp
from dynamical_system import replicator_niche_overlap
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(layout="wide")
st.title("Replicator equation")

st.header("Exponential growth population")


r1 = st.sidebar.slider("Intrinsic growth rate of type 1", 0.0, 3.0, value=2.0, step=0.1)
r2 = st.sidebar.slider("Intrinsic growth rate of type 2", 0.0, 3.0, value=2.2, step=0.1)
r3 = st.sidebar.slider("Intrinsic growth rate of type 3", 0.0, 3.0, value=1.8, step=0.1)

init = np.array([0.1 / 0.3, 0.1 / 0.3, 0.1 / 0.3, 0.3])
pars = np.array([r1, r2, r3, 0])
exp_sol = solve_ivp(replicator_niche_overlap, (0, 50), init, args=(pars,))

df_exp = pd.DataFrame(
    dict(
        time=exp_sol.t,
        freq_type_1=exp_sol.y[0, :],
        freq_type_2=exp_sol.y[1, :],
        freq_type_3=exp_sol.y[2, :],
        total=exp_sol.y[3, :],
    )
)

df_exp["density_type_1"] = df_exp.freq_type_1 * df_exp.total
df_exp["density_type_2"] = df_exp.freq_type_2 * df_exp.total
df_exp["density_type_3"] = df_exp.freq_type_3 * df_exp.total

col1, col2 = st.columns(2)

with col1:
    st.altair_chart(
        alt.Chart(df_exp)
        .transform_fold(["feq_type_1", "freq_type_2", "freq_type_3"])
        .mark_line()
        .encode(x="time", y="value:Q", color="key:N")
    )

with col2:
    st.altair_chart(
        alt.Chart(df_exp)
        .transform_fold(["denisty_type_1", "density_type_2", "density_type_3", "total"])
        .mark_line()
        .encode(x="time", y=alt.Y("value:Q", scale=alt.Scale(type="log")), color="key:N")
    )

st.header("Logistic growth without niche difference")

alpha = st.sidebar.slider("Competition coefficient", 0.0, 1.0, value=0.2, step=0.1)

pars_logi = np.array([r1, r2, r3, alpha])
logi_sol = solve_ivp(replicator_niche_overlap, (0, 50), init, args=(pars_logi,))

df_logi = pd.DataFrame(
    dict(
        time=logi_sol.t,
        freq_type_1=logi_sol.y[0, :],
        freq_type_2=logi_sol.y[1, :],
        freq_type_3=logi_sol.y[2, :],
        total=logi_sol.y[3, :],
    )
)

st.altair_chart(
    alt.Chart(df_logi)
    .transform_fold(["feq_type_1", "freq_type_2", "freq_type_3"])
    .mark_line()
    .encode(x="time", y="value:Q", color="key:N")
)
