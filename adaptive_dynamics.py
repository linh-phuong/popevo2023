import streamlit as st
from dynamical_system import invasion_fitness
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Adaptive dynamics")

zlist = np.linspace(0, 10, 50)

zm = st.sidebar.number_input("Mutant trait value", 0.0, 10.0, value=2.0, step=1e-4, format="%.4f")
alpha = 0.4

inv_fitness = invasion_fitness(zlist, zm, alpha=alpha)

X, Y = np.meshgrid(zlist, zlist)
inv_fitness3D = invasion_fitness(X, Y, alpha=alpha)

fig = px.line(
    x=zlist, y=inv_fitness, labels={"x": "Resident trait value (z)", "y": "Invasion fitness"}
)
fig.add_vline(x=zm, line_dash="dash")
fig.add_hline(y=0, line_dash="dash")
st.plotly_chart(fig)

fig = go.Figure(data=[go.Surface(z=inv_fitness3D, colorscale="RdBu")])
st.plotly_chart(fig)
