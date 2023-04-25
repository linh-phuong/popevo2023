import streamlit as st
from dynamical_system import invasion_fitness
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tools import plot_3D_invfitness

st.set_page_config(layout="wide")
st.title("Adaptive dynamics")

st.subheader("Invasion process")


zlist = np.linspace(0, 10, 50)
alpha = 0.4

X, Y = np.meshgrid(zlist, zlist)
inv_fitness3D = invasion_fitness(X, Y, alpha=alpha)

col1, col2 = st.columns(2, gap="large")

inv_vid = []
for z_val in np.linspace(2, zlist[-1], 20):
    inv_vid.append(invasion_fitness(zlist, z_val, alpha=alpha))
frame_dat = []
with col1:
    vid = go.Figure(
        data=[
            go.Line(x=zlist, y=invasion_fitness(zlist, 2, alpha=alpha), name="invasion fitness"),
            go.Line(
                x=zlist,
                y=np.zeros(len(zlist)),
                line=dict(color="black", width=1, dash="dash"),
                name="Invasion threshold",
            ),
            go.Line(
                x=[2] * 10,
                y=np.linspace(-10, 10, 10),
                line=dict(color="black", width=3, dash="dash"),
                name="Resident trait value",
            ),
        ],
        layout=go.Layout(
            xaxis=dict(range=[0, 10], autorange=False),
            yaxis=dict(range=[-10, 10], autorange=False),
            updatemenus=[
                dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])
            ],
        ),
        frames=[
            go.Frame(
                data=[
                    go.Line(x=zlist, y=i),
                    go.Line(
                        x=zlist,
                        y=np.zeros(len(zlist)),
                        line=dict(color="black", width=1, dash="dash"),
                    ),
                    go.Line(
                        x=[z_val] * 10,
                        y=np.linspace(-10, 10, 10),
                        line=dict(color="black", width=3, dash="dash"),
                    ),
                ]
            )
            for i, z_val in zip(inv_vid, np.linspace(2, zlist[-1], 20))
        ],
    )
    st.plotly_chart(vid)
with col2:
    vid2 = go.Figure(
        data=go.Contour(x=zlist, y=zlist, z=inv_fitness3D),
        layout=go.Layout(autosize=False, width=700, height=700),
    )
    st.plotly_chart(vid2)

zm = st.sidebar.number_input("Mutant trait value", 0.0, 10.0, value=2.0, step=1e-4, format="%.4f")

inv_fitness = invasion_fitness(zlist, zm, alpha=alpha)


col1, col2 = st.columns(2, gap="large")

with col1:
    fig = px.line(
        x=zlist, y=inv_fitness, labels={"x": "Mutant trait value (z)", "y": "Invasion fitness"}
    )
    fig.add_vline(x=zm, line_dash="dash")
    fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig)
with col2:
    st.plotly_chart(plot_3D_invfitness(zlist, inv_fitness3D, zm))
