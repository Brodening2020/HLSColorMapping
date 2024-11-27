import numpy as np
import cv2
import math
import plotly.graph_objects as go
from jinja2 import Template

imgpath=r"./TestImages/pic4.jpg"
input_template_path = r"./templates/result_template.html"
output_html_path=r"./templates/{}.html".format(imgpath.split("/")[-1].split(".")[0])

resize_height=1000
#HLS毎に何階級に分けるか
bins=(20, 10, 10)
threshold=1
threshold2=None

#度数何個ごとに点の大きさ変えるか
#pixelの数なので画像全体に占める割合%で、最大5%として計算してみる
approx_size=resize_height**2
count_threshold=np.dot(approx_size, [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05])
pointsizes_go=np.array([3, 5, 10, 15, 20, 30, 35])

#グラフの描画領域の大きさpx
graph_height=800
graph_width=800


# 画像をクリップボードから読み込み、リサイズし、RGB版とHLS版を返す
def read_image(imgpath, resize_height):
    img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    img = cv2.resize(img, (resize_height*width//height, resize_height))
    im = np.array(img)
    #HLSに変換
    im_hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS_FULL).reshape(-1, 3).astype(float)
    return im_hls, img

#近傍の色をまとめて、threshold個以上含まれる色のみ抽出する
#binsは、何階級に分けるかを指定している これにより近傍色まとまる
def histogram(im_hls, bins, threshold, threshold2):
    #HLSの各成分について一定階級ごとに分割、ヒストグラムで分布計算
    #bin_edgeの代わりにrangeで指定
    #aはhistogram行列、bにはedgeが返ってくる
    a, b = np.histogramdd(im_hls, bins=bins, range=((0, 255), (0, 255), (0, 255)))
    #HLSの各成分について隣接するbin_edgeの平均値（階級値）計算
    c = [((b[i][1:] + b[i][:-1])/2).round().astype(int) for i in range(3)]
    #ヒストグラムaでの度数が1個以上かつ{threshold2}個以下の部分を抽出、aにおけるその部分のリスト番号取得
    if threshold2 is None:
        idx = np.where(a>=threshold)
    else:
        idx = np.where((a>=threshold)&(a<=threshold2))
    #抽出した部分の個数
    count = a[idx]
    #抽出部分に対応する階級値を抽出　HSL各成分ごと
    hls = np.array([c[i][idx[i]].astype(float) for i in range(3)])
    #近傍色まとめたのでRGB（%）にも変換して返す
    rgb = cv2.cvtColor(hls.T[None].astype(np.uint8), cv2.COLOR_HLS2RGB_FULL)[0].astype(float)
    rgb_percent=rgb/255
    return hls, rgb, rgb_percent, count

#HLSを％に直してクリスタの正三角形カラーサークルに写像
def hls2Triangle(h, l, s):
    #opencvはクズなので色相Hを0-255で返します許さん
    h=h*360/255
    l=l/255
    s=s/255
    s_tri=s*np.minimum(l, 1-l)*math.sqrt(3)
    return h, l, s, s_tri

#度数に応じて点の描画サイズを変える関数
def pointsize(count, count_threshold, pointsizes):
    sizes=np.zeros(len(count))
    for i, c in enumerate(count):
        if c<=count_threshold[-1]:
            thres_id=np.where(count_threshold>=c)[0][0]
        else:
            thres_id=len(count_threshold)-1
        sizes[i]=pointsizes[thres_id]
    return sizes

def plot_go(hls, rgb, count, title):
    #各色をHLSでプロット *で渡すことでリストの各要素を引数として受け取ってくれる
    h, l, s, s_tri = hls2Triangle(*hls)
    #各色バブルの大きさを取得
    size=pointsize(count, count_threshold, pointsizes_go)
    #rgbで指定した点の色をplotlyがわかるフォーマットに書き換え
    rgb_array=['rgb({},{},{})'.format(r, g, b) for r, g, b in rgb]
    #マウスをホバーしたときに各点で表示するテキストのリスト
    hovertext=["Count:{} H:{} L:{} S:{}".format(int(count[i]), int(h[i]), int(l[i]*100), int(s[i]*100))
               for i in range(len(h))]
    #plotlyでバブルチャート作成　legend出すと重いので必ずオフ
    trace = go.Scatter3d(x=l, y=s_tri, z=h, text=hovertext, mode='markers', 
                        marker=dict(size=size, color=rgb_array, opacity=1.0), 
                        hoverinfo="text",
                        showlegend=False)
    layout = go.Layout(
        height=graph_height,
        width=graph_width,
        title={
            'text': title,
            "x":0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        scene=dict(
            xaxis=dict(dtick=0.25, range=[1.1, -0.1], title='輝度L'),
            yaxis=dict(dtick=0.25, range=[-0.1, 1.1], title='彩度S_TRI'),
            zaxis=dict(dtick=30, range=[0, 370], title='色相H')
            ),
        font=dict(family="Arial")
    )
    fig = go.Figure(data=[trace], layout=layout)

    #クリスタのLS三角形領域をプロット　ホバー情報は表示しない
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3)/2, 0]
    tri_z=[0, 0, 0, 0]
    fig.add_trace(go.Scatter3d(x=tri_x, y=tri_y, z=tri_z, mode="lines", 
                               line=dict(color='blue'),
                               hoverinfo="skip",
                               showlegend=False))
    #fig.show()
    return fig

def html_output(fig):
    plotly_jinja_data = {"fig":fig.to_html(full_html=False)}
    #consider also defining the include_plotlyjs parameter to point to an external Plotly.js as described above
    with open(output_html_path, "w", encoding="utf-8") as output_file:
        with open(input_template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render(plotly_jinja_data))

im_hls, img=read_image(imgpath, resize_height)
hls, rgb, rgb_percent, count=histogram(im_hls, bins, threshold, threshold2)
print(count.shape)
print(hls.shape)
print(rgb.shape)
title=imgpath.split("/")[-1]
fig=plot_go(hls, rgb, count, title)
html_output(fig)