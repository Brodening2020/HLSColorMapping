import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd

"""
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': '3d'})

    #クリスタのLS三角形領域をプロット
    tri_points = np.array([[0, 0, 0], [0, 1, 0], [np.sqrt(3)/2, 0.5, 0], [0, 0, 0]])
    ax.plot(tri_points[:, 0], tri_points[:, 1], color='blue', linewidth=2)

    #各色をHLSでプロット
    h, l, s, s_tri = hls2Triangle(*hls)

    sc=ax.scatter(s_tri, l, h, s=pointsize(count, count_threshold, pointsizes), c=rgb, alpha=1)

    plt.xticks(np.arange(0, 1.25, 0.25))
    plt.yticks(np.arange(0, 1.25, 0.25))
    ax.set_xlabel('S_TRI')
    ax.set_ylabel('L')
    ax.set_zlabel('H')
    ax.set_zlim(0)
"""

#df = px.data.gapminder()
#pxだとscatterに渡すデータはすべてpandasのdf　そうじゃなくても内部でpandasのdfに変換される
#np.arrayはplotlyは受け取らない　pandasのdfで全部まとめるならOK
import plotly.graph_objs as go
import numpy as np

# データの作成
x = np.random.rand(5)
y = np.random.rand(5)
z = np.random.rand(5)
colors = np.random.rand(5, 3)  # 各点のRGB値

# RGB値を16進数に変換
color_hex = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]
print(color_hex)

# 3D散布図を作成
trace = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=color_hex  # 点の色を指定
    )
)

layout = go.Layout(
    scene=dict(
        xaxis_title='X軸',
        yaxis_title='Y軸',
        zaxis_title='Z軸'
    )
)

fig = go.Figure(data=[trace], layout=layout)
fig.show()
