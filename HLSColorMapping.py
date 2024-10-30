import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import math
from PIL import ImageGrab

#imgpath=r"./test.png"
imgpath=r"./Test3.png"
resize_height=1000
#HLS毎に何階級に分けるか
bins=(20, 10, 10)
threshold=1
threshold2=None

#度数何個ごとに点の大きさ変えるか
#pixelの数なので画像全体に占める割合%で、最大5%として計算してみる
approx_size=resize_height**2
count_threshold=np.dot(approx_size, [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05])
pointsizes=np.array([1, 5, 10, 50, 100, 200, 500])

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
    rgb = cv2.cvtColor(hls.T[None].astype(np.uint8), cv2.COLOR_HLS2RGB_FULL)[0].astype(float)/255
    return hls, rgb, count

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


def plot(hls, rgb, count):
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
    
    #マウスホバーしたときにHLSを表示
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        i = ind["ind"][0]
        pos = sc.get_offsets()[i]
        annot.xy = pos
        text = "H:"+str(int(h[i]))+" L:"+str(int(l[i]*100))+" S:"+str(int(s[i]*100))+" Count:"+str(int(count[i]))
        annot.set_text(text)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                   annot.set_visible(False)
                   fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    plt.tight_layout()
    plt.show()
    return fig, ax

im_hls, img=read_image(imgpath, resize_height)
hls, rgb, count=histogram(im_hls, bins, threshold, threshold2)
print(count.shape)
print(hls.shape)
fig, ax=plot(hls, rgb, count)