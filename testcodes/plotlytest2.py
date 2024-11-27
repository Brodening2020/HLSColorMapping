import plotly.graph_objs as go
import numpy as np

# データを作成
x = np.random.rand(50)
y = np.random.rand(50)
z = np.random.rand(50)

# 3D散布図を作成
trace = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color='blue'  # 色を指定
    )
)

# レイアウトを設定し、目盛り幅を変更
layout = go.Layout(
    scene=dict(
        xaxis=dict(
            dtick=0.1,  # X軸の目盛り幅を指定
            range=[0, 1],  # X軸の範囲を指定
            title='X軸'
        ),
        yaxis=dict(
            dtick=0.1,  # Y軸の目盛り幅を指定
            range=[0, 1],  # Y軸の範囲を指定
            title='Y軸'
        ),
        zaxis=dict(
            dtick=1,  # Z軸の目盛り幅を指定
            range=[0, 10],  # Z軸の範囲を指定
            title='Z軸'
        )
    )
)

fig = go.Figure(data=[trace], layout=layout)
fig.show()
