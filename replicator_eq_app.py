import streamlit as st
from scipy.integrate import solve_ivp
from dynamical_system import replicator_niche_overlap, replicator
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(layout="wide")
st.title("Replicator equation")

st.header("Exponential growth population")

st.markdown(
    r"""
We consider three types with three different growth rates $r_1, r_2, r_3$
The system of ODEs that describes the density dynamics are

$$
\begin{align*}
& \frac{dN_1}{dt} =  r_1 N_1 \\
& \frac{dN_2}{dt} = r_2 N_2 \\
& \frac{dN_3}{dt} = r_3 N_3 \\
& \frac{dN_{total}}{dt} = \bar{r} N_{total}
\end{align*}
$$

where $\bar{r} = r_1 \frac{N_1}{N_{total}} + r_2 \frac{N_2}{N_{total}} + r_3 \frac{N_3}{N_{total}}$ is the average growth rate

From Lion (2018), we could work out the dynamics for the frequency of the three types as followed

$$
\begin{align*}
& \frac{df_1}{dt} = (r_1 - \bar{r}) f_1 \\
& \frac{df_2}{dt} = (r_2 - \bar{r}) f_2 \\
& \frac{df_3}{dt} = (r_3 - \bar{r}) f_3 \\
\end{align*}
$$

and $f_i = \frac{N_i}{N_{total}}$ is the frequency of type i
"""
)

st.write(
    "Try to vary the growth rates of the different types, what changes? Can three species coexist?"
)

st.write("")

r1 = st.sidebar.slider("Intrinsic growth rate of type 1", 0.0, 3.0, value=2.0, step=0.1)
r2 = st.sidebar.slider("Intrinsic growth rate of type 2", 0.0, 3.0, value=2.2, step=0.1)
r3 = st.sidebar.slider("Intrinsic growth rate of type 3", 0.0, 3.0, value=1.8, step=0.1)

ntotal = 0.3
init = np.array([0.1 / ntotal, 0.1 / ntotal, 0.1 / ntotal, ntotal])

pars = np.array([r1, r2, r3, 0])
exp_sol = solve_ivp(
    replicator_niche_overlap, (0, 50), init, args=(pars,), t_eval=np.arange(0, 50, 0.5)
)

df_exp = pd.DataFrame(
    dict(
        time=exp_sol.t,
        type_1=exp_sol.y[0, :],
        type_2=exp_sol.y[1, :],
        type_3=exp_sol.y[2, :],
        total=exp_sol.y[3, :],
    )
)


col1, col2 = st.columns(2)

with col1:
    st.altair_chart(
        alt.Chart(df_exp)
        .transform_fold(["type_1", "type_2", "type_3"])
        .mark_line()
        .encode(
            x="time",
            y=alt.Y("value:Q", title="Population frequency"),
            color=alt.Color("key:N", legend=alt.Legend(title="Frequency")),
        )
    )

with col2:
    st.altair_chart(
        alt.Chart(df_exp)
        .transform_fold(["total"])
        .mark_line()
        .encode(
            x="time",
            y=alt.Y("value:Q", scale=alt.Scale(type="log"), title="Total population density"),
            color=alt.value("green"),
        )
    )

st.header("Logistic growth without niche difference")

st.markdown(
    r"""
We still consider three types as in previous section but now we add the same competition among types.

The dynamics of population density are now as followed

$$
\begin{align*}
& \frac{dN_1}{dt} =  (r_1 - \alpha N_{total}) N_1 \\
& \frac{dN_2}{dt} = (r_2  - \alpha N_{total}) N_2\\
& \frac{dN_3}{dt} = (r_3  - \alpha N_{total}) N_3 \\
& \frac{dN_{total}}{dt} = \bar{r} N_{total}
\end{align*}
$$

Now, $\bar{r} = (r_1 - \alpha N_{total}) \frac{N_1}{N_{total}} + 
(r_2 - \alpha N_{total}) \frac{N_2}{N_{total}} + (r_3 - \alpha N_{total}) \frac{N_3}{N_{total}}$

Again, the dynamics for the frequency are

$$
\begin{align*}
& \frac{df_1}{dt} =  (r_1 - \bar{r}) f_1 \\
& \frac{df_2}{dt} = (r_2   - \bar{r}) f_2\\
& \frac{df_3}{dt} = (r_3   - \bar{r}) f_3 \\
\end{align*}
$$

"""
)

st.write(
    r"""
Now notice that the formular for the dynamics of frequency is the same 
but $\bar{r}$ is different from the exponential example
"""
)

st.write(
    "Try to vary the competition coefficient, what is changing? Do you see the differerence in the density dynamics?"
)
alpha = st.slider("Competition coefficient", 0.0, 1.0, value=0.2, step=0.1)

pars_logi = np.array([r1, r2, r3, alpha])

logi_sol = solve_ivp(
    replicator_niche_overlap, (0, 50), init, args=(pars_logi,), t_eval=np.arange(0, 50, 0.5)
)

df_logi = pd.DataFrame(
    dict(
        time=logi_sol.t,
        type_1=logi_sol.y[0, :],
        type_2=logi_sol.y[1, :],
        type_3=logi_sol.y[2, :],
        total=logi_sol.y[3, :],
    )
)


col3, col4 = st.columns(2)

with col3:
    st.altair_chart(
        alt.Chart(df_logi)
        .transform_fold(["type_1", "type_2", "type_3"])
        .mark_line()
        .encode(
            x="time", y="value:Q", color=alt.Color("key:N", legend=alt.Legend(title="Frequency"))
        )
    )

with col4:
    st.altair_chart(
        alt.Chart(df_logi)
        .transform_fold(["total"])
        .mark_line()
        .encode(x="time", y="value:Q", color=alt.value("green"))
    )

st.header("Asymetric competitions create niche difference")

a11 = st.slider("Intraspecific competition of type 1", 0.0, 2.0, value=1.0, step=0.1)
a22 = st.slider("Intraspecific competition of type 2", 0.0, 2.0, value=1.0, step=0.1)
a33 = st.slider("Intraspecific competition of type 3", 0.0, 2.0, value=1.0, step=0.1)
tmax = st.slider("Integrate time", 50.0, 200.0, value=50.0, step=0.5)

pars_nichediff = np.array([r1, r2, r3, a11, 0.8, 0.8, 0.8, a22, 0.8, 0.8, 0.8, a33])

nichediff_sol = solve_ivp(
    replicator, (0, tmax), init, t_eval=np.arange(0, tmax, 0.1), args=(pars_nichediff,)
)


df_nd = pd.DataFrame(
    dict(
        time=nichediff_sol.t,
        freq_type_1=nichediff_sol.y[0, :],
        freq_type_2=nichediff_sol.y[1, :],
        freq_type_3=nichediff_sol.y[2, :],
        total=nichediff_sol.y[3, :],
    )
)


st.altair_chart(
    alt.Chart(df_nd)
    .transform_fold(["freq_type_1", "freq_type_2", "freq_type_3"])
    .mark_line()
    .encode(x="time", y="value:Q", color="key:N")
)

st.write(df_nd.tail(5))
