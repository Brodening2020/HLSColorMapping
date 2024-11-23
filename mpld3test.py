import plotly.graph_objs as go
import numpy as np

x = np.linspace(-5, 5, 100)
y = np.sin(x)
z = np.cos(x)

trace = go.Scatter3d(x=x, y=y, z=z, mode='lines')

layout = go.Layout(
    scene=dict(
        xaxis_title='X軸',
        yaxis_title='Y軸',
        zaxis_title='Z軸'
    )
)

fig = go.Figure(data=[trace], layout=layout)
fig.write_html('3d_plot.html')
