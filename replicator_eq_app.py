import streamlit as st
from scipy.integrate import solve_ivp
from dynamical_system import replicator_niche_overlap, replicator
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(layout="wide")
st.title("Replicator equation")

st.header("Exponential growth population")


r1 = st.sidebar.slider("Intrinsic growth rate of type 1",
                       0.0, 3.0, value=2.0, step=0.1)
r2 = st.sidebar.slider("Intrinsic growth rate of type 2",
                       0.0, 3.0, value=2.2, step=0.1)
r3 = st.sidebar.slider("Intrinsic growth rate of type 3",
                       0.0, 3.0, value=1.8, step=0.1)

ntotal = 0.3
init = np.array([0.1 / ntotal, 0.1 / ntotal, 0.1 / ntotal, ntotal])

pars = np.array([r1, r2, r3, 0])
exp_sol = solve_ivp(replicator_niche_overlap, (0, 50), init,
                    args=(pars,), t_eval=np.arange(0, 50, 0.5))

df_exp = pd.DataFrame(
    dict(
        time=exp_sol.t,
        freq_type_1=exp_sol.y[0, :],
        freq_type_2=exp_sol.y[1, :],
        freq_type_3=exp_sol.y[2, :],
        total=exp_sol.y[3, :],
    )
)


col1, col2 = st.columns(2)

with col1:
    st.altair_chart(
        alt.Chart(df_exp)
        .transform_fold(["freq_type_1", "freq_type_2", "freq_type_3"])
        .mark_line()
        .encode(x="time", y="value:Q", color="key:N")
    )

with col2:
    st.altair_chart(
        alt.Chart(df_exp)
        .transform_fold(["total"])
        .mark_line()
        .encode(x="time", y=alt.Y("value:Q", scale=alt.Scale(type="log")), color=alt.value("green"))
    )

st.header("Logistic growth without niche difference")

alpha = st.sidebar.slider("Competition coefficient",
                          0.0, 1.0, value=0.2, step=0.1)

pars_logi = np.array([r1, r2, r3, alpha])

logi_sol = solve_ivp(replicator_niche_overlap,
                     (0, 50), init, args=(pars_logi,), t_eval=np.arange(0, 50, 0.5))

df_logi = pd.DataFrame(
    dict(
        time=logi_sol.t,
        freq_type_1=logi_sol.y[0, :],
        freq_type_2=logi_sol.y[1, :],
        freq_type_3=logi_sol.y[2, :],
        total=logi_sol.y[3, :],
    )
)


col3, col4 = st.columns(2)

with col3:
    st.altair_chart(
        alt.Chart(df_logi)
        .transform_fold(["freq_type_1", "freq_type_2", "freq_type_3"])
        .mark_line()
        .encode(x="time", y="value:Q", color="key:N")
    )

with col4:
    st.altair_chart(
        alt.Chart(df_logi)
        .transform_fold(["total"])
        .mark_line()
        .encode(x="time", y="value:Q", color=alt.value("green"))
    )

st.header("Asymetric competitions create niche difference")

a11 = st.sidebar.slider(
    "Intraspecific competition of type 1", 0., 2., value=1., step=0.1)
a22 = st.sidebar.slider(
    "Intraspecific competition of type 2", 0., 2., value=1., step=0.1)
a33 = st.sidebar.slider(
    "Intraspecific competition of type 3", 0., 2., value=1., step=0.1)

pars_nichediff = np.array(
    [r1, r2, r3, a11, 0.8, 0.8, 0.8, a22, 0.8, 0.8, 0.8, a33])

nichediff_sol = solve_ivp(replicator, (0, 50), init,
                          t_eval=np.arange(0, 50, 0.5), args=(pars_nichediff,))


df_nd = pd.DataFrame(dict(time=nichediff_sol.t,
                     freq_type_1=nichediff_sol.y[0, :], freq_type_2=nichediff_sol.y[1, :], freq_type_3=nichediff_sol.y[2, :], total=nichediff_sol.y[3, :]))


st.altair_chart(alt.Chart(df_nd).transform_fold(
    ["freq_type_1", "freq_type_2", "freq_type_3"]).mark_line().encode(x='time', y='value:Q', color='key:N'))

st.write(df_nd.head(10))
