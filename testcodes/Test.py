import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ウインドウとグラフを同時に作成
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection' : '3d'})
#numpyを使ってXYZの値を設定
x = np.random.rand(200,1) 
y = np.random.rand(200,1)
z = np.random.rand(200,1)

#散布図の作成
ax.scatter(x,y,z,s=40,c="red")

#描画
plt.show()