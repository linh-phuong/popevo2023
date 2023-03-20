import streamlit as st
import numpy as np
from tools import integrate
from dynamical_system import structural_population
import pandas as pd
import altair as alt

st.set_page_config(layout="wide")

st.title("Leslie matrix for structural population")

st.header("Matrix operation and intergration of the population dynamics")
st.markdown(
    r"""
This file give the example of a population with 4 age classes: age 1, age 2, age 3 and age 4

The dynamics of the population can be written in matrix form:
$$
\begin{align*}
N_1(t + 1) \\ 
N_2(t + 1) \\ 
N_3(t + 1) \\ 
N_4(t + 1)
\end{align*}= 
\begin{pmatrix}
F_1 & F_2 & F_3 & F_4 \\
s_1 & 0 & 0 & 0 \\
0 & s_2 & 0 & 0 \\
0 & 0 & s_3 & 0
\end{pmatrix}
\begin{pmatrix}
N_1(t) \\ 
N_2(t) \\ 
N_3(t) \\ 
N_4(t)
\end{pmatrix}
$$
where $F_1$, $F_2$, $F_3$, and $F_4$ are the fecundity that corresponds to age class 1, 2, 3, and 4. 
$s_1$, $s_2$, $s_3$, are the survival rate of age class 1, 2, and 3. 
Since the maximum age of this population is 4, survival rate of the age class 4 is 0.

Check it your self to see if this result is correct from the matrix multiplication above

$$
\begin{align*}
& N_1(t+1) = F_1 N_1(t) + F_2 N_2(t) + F_3 N_3(t) + F_4 N_4(t) \\
& N_2(t+1) = s_1 N_1(t) \\
& N_3(t+1) = s_2 N_2(t) \\
& N_4(t+1) = s_3 N_3(t)
\end{align*}
$$

In the following graphs, we always assume that only individuals from age 3 and age 4 can reproduce, 
there fore $F_1 = F_2 = 0$

"""
)

tmax = st.sidebar.number_input("Intergrate time", 2, 100, value=40, step=1)
a1 = st.sidebar.slider("Initial population age 1", 0, 150, value=100, step=1)
a2 = st.sidebar.slider("Initial population age 2", 0, 150, value=0, step=1)
a3 = st.sidebar.slider("Initial population age 3", 0, 150, value=0, step=1)
a4 = st.sidebar.slider("Initial population age 4", 0, 150, value=0, step=1)
f3 = st.sidebar.slider("Fecundity of age class 3", 0.0, 10.0, value=2.0, step=0.5)
f4 = st.sidebar.slider("Fecundity of age class 4", 0.0, 10.0, value=4.0, step=0.5)
s1 = st.sidebar.slider("Survival rate of age class 1", 0.0, 1.0, value=0.5, step=0.1)
s2 = st.sidebar.slider("Survival rate of age class 2", 0.0, 1.0, value=1.0, step=0.1)
s3 = st.sidebar.slider("Survival rate of age class 3", 0.0, 1.0, value=0.5, step=0.1)

init_pop = np.array([a1, a2, a3, a4])
L_matrix = np.array([[0, 0, f3, f4], [s1, 0, 0, 0], [0, s2, 0, 0], [0, 0, s3, 0]])
time, pop_dynamics = integrate(structural_population, init_pop, tmax, cg=L_matrix)
pop_dynamics = np.array(pop_dynamics)

df = pd.DataFrame(
    dict(
        time=time,
        age_1=pop_dynamics[:, 0],
        age_2=pop_dynamics[:, 1],
        age_3=pop_dynamics[:, 2],
        age_4=pop_dynamics[:, 3],
    )
)

df["total"] = df[["age_1", "age_2", "age_3", "age_4"]].sum(axis=1)

st.write(
    """The following table shows the first ten intergration time step. 
    Spend sometimes to intergrate the dynamics yourself and check the values"""
)

st.write(df.head(10))

st.write("")

df["fraction_age_1"] = df.age_1 / df.total
df["fraction_age_2"] = df.age_2 / df.total
df["fraction_age_3"] = df.age_3 / df.total
df["fraction_age_4"] = df.age_4 / df.total

st.header("Graphical illustration of the population dynamics")

st.write("The following graph illustrates the population dynamics")

col1, col2 = st.columns(2)

with col1:
    st.altair_chart(
        alt.Chart(df)
        .transform_fold(["age_1", "age_2", "age_3", "age_4", "total"])
        .transform_filter(alt.datum.value > 0)
        .mark_point()
        .encode(
            x="time",
            y=alt.Y(
                "value:Q",
                scale=alt.Scale(type="log"),
                title="Population density (log scale)",
            ),
            color="key:N",
        )
    )

with col2:
    st.altair_chart(
        alt.Chart(df)
        .transform_fold(["fraction_age_1", "fraction_age_2", "fraction_age_3", "fraction_age_4"])
        .transform_filter(alt.datum.value > 0)
        .mark_point()
        .encode(
            x="time",
            y=alt.Y(
                "value:Q",
                title="Fraction of population",
            ),
            color="key:N",
        )
    )

st.header("Percapital growth rate and eigenvalue")

st.markdown(
    r"""
The percapital growth rate of the population at each year is
$$\lambda(t) = \frac{N(t)}{N(t - 1)}$$
"""
)

lbd = df["total"].iloc[1:].values / df["total"].iloc[:-1].values
leading_lbd = np.linalg.eigvals(L_matrix).max().real
lbd_df = pd.DataFrame(
    {
        "lambda_t": lbd,
        "Time": time[1:],
        "leading_eigenval": [leading_lbd] * len(time[1:]),
    }
)
tt = alt.TitleParams(r"lambda(t) and leading eigenvalue of Leslie matrix", anchor="middle")
l1 = (
    alt.Chart(lbd_df, title=tt)
    .mark_point()
    .encode(x="Time", y=alt.Y("lambda_t", title=r"lambda(t)"))
)
l2 = alt.Chart(lbd_df).mark_line().encode(x="Time", y="leading_eigenval")
st.altair_chart(l1 + l2)

st.header("Survivalship, life expectancy and R0")
st.write(r"The following table shows the survivalship until age x, life expectancy and $R_0$.")

life_expt = np.array([1.0, s1, s1 * s2, s1 * s2 * s3])
st.write(
    pd.DataFrame(
        dict(
            Survivalship_until_age_1=[life_expt[0]],
            Survivalship_until_age_2=[life_expt[1]],
            Survivalship_until_age_3=[life_expt[2]],
            Survivalship_until_age_4=[life_expt[3]],
            Life_expectancy=[life_expt.sum()],
            R0=[np.sum([0, 0, f3 * s1 * s2, f4 * s1 * s2 * s3])],
        )
    )
)

st.header("Suggestions")
st.write(
    r"""
Vary the parameters to see how the population size change. Note that if the iteration time is too small, 
the percapital growth rate of the population $\lambda(t)$ has not converted to the eigen value

Some suggestions to play around with the parameters:

_ Keep other parameters fixed, vary the "Integrate time" and verify the result 
of the first few steps of the iteration by yourself

_ Keep other parameters fixed, vary the fecundity F3 or F4. 
Does the structure of the population change? Does the R0 change? What happens to the population if the $R_0$ < 1

_ Keep other parameters fixed, vary the initial population values. Does the structure of the population change?

"""
)
