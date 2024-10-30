import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import math
from PIL import ImageGrab

# 画像をクリップボードから読み込み、リサイズし、RGB版とHLS版を返す
def read_image(imgpath, height=256):
    img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
    im = np.array(img)
    #HLSに変換
    im_hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS_FULL).reshape(-1, 3).astype(float)
    return im_hls, img

#近傍の色をまとめて、threshold個以上含まれる色のみ抽出する
#binsは、何階級に分けるかを指定している これにより近傍色まとまる
def histogram(im_hls, bins=(64, 32, 16), threshold=1, threshold2=None):
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
    #抽出した部分の全個数カウント
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
    return h, l, s_tri

def plot(hls, rgb):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    h, l, s_tri = hls2Triangle(*hls)
    ax.scatter(s_tri, l, 0, c=rgb)

    plt.xticks(np.arange(0, 1.25, 0.25))
    plt.yticks(np.arange(0, 1.25, 0.25))
    ax.set_zlim(0)
    ax.set_xlabel('S_TRI')
    ax.set_ylabel('L')
    ax.set_zlabel('H')

    # 三角形の頂点を定義
    points = np.array([[0, 0, 0], [0, 1, 0], [np.sqrt(3)/2, 0.5, 0], [0, 0, 0]])

    # 三角形を描画
    ax.plot(points[:, 0], points[:, 1], 'bo-')  # 'bo-' は青い点と線を意味します

    plt.axis('equal') # 軸のスケーリングを等しくする
    plt.show()
    return fig, ax


imgpath=r"./test.png"
im_hls, img=read_image(imgpath)
hls, rgb, count=histogram(im_hls)
plot(hls, rgb)





"""
# HLSから球内に写像する
def hls2xyz(h, l, s):
    h *= 2 * np.pi / 255
    l *= 2 / 255
    l -= 1
    s *= 1 / 255

    x = s * np.cos(h) * np.cos(np.arcsin(l))
    y = s * np.sin(h) * np.cos(np.arcsin(l))
    z = l
    return x, y, z

# 球内の座標に対応するRGBを計算する
def xyz2rgb(x, y, z, sat=1):
    h = (np.arctan2(-y, -x) + np.pi) * 255 / 2 / np.pi
    l = (z + 1) / 2 * 255
    s = np.full_like(z, 255*sat)
    rgb = cv2.cvtColor(np.c_[h, l, s][None].astype(np.uint8), cv2.COLOR_HLS2RGB_FULL)[0].astype(float) / 255
    return rgb

# プロットを行う
def plot(hls, rgb, size, sphere=True, sphere_type=1, size_max=1000):
    
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.set_facecolor((0.25, 0.25, 0.25))
    
    if sphere:
        th = np.r_[0:2*np.pi:100j]
        if sphere_type in (1, "both"):
            for i in np.r_[-.95:.95:10j]:
                r = np.cos(np.arcsin(i))
                circle = np.c_[r*np.cos(th), r*np.sin(th), np.full(100, i)].T
                rgb_ = xyz2rgb(*circle)
                ax.scatter(*circle, c=rgb_, alpha=.1, s=1)
        if sphere_type in (2, "both"):
            a = np.c_[np.zeros_like(th), np.cos(th), np.sin(th)]
            for i in np.r_[0:np.pi:np.pi/6]:
                circle = (a @ np.c_[[np.cos(i), -np.sin(i), 0], [np.sin(i), np.cos(i), 0], [0, 0, 1]]).T
                rgb_ = xyz2rgb(*circle)
                ax.scatter(*circle, c=rgb_, alpha=.1, s=1)
    
    x, y, z = hls2xyz(*hls)
    ax.scatter(x, y, z, s=np.clip(size, 0, size_max), alpha=.7, c=rgb)
        
    ax.set_proj_type('ortho')
    ax.axis('equal')
    ax.axis('off')
    plt.show()
    return fig, ax
    """
