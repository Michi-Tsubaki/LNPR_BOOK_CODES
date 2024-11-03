#!/usr/bin/env python3

#確率ロボティクス第2章　確率統計の復習
##「多次元ガウス分布」

#ライブラリのインポート
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sci

##データ"sensor_data_700.txt"のインポート
data = pd.read_csv("sensor_data_700.txt", delimiter=" ",
                   header=None, names=("data","time","ir","lidar"))

d = data[(data["time"]<160000) & (data["time"] >= 120000)] #12時から16時までのデータを抽出
d = d.loc[:,["ir","lidar"]] #LiDARの値と光センサの値
##描画
sns.jointplot(x="ir", y="lidar", data=d, kind="kde", fill=True, thresh=0)
plt.show()
##平均・共分散行列
print("sensor_data_700.txt")
print("平均")
mean = d.mean()
print(mean)
print("共分散行列")
cov = d.cov()
print(cov)

#多次元ガウス分布でモデル化して描画
irlidar = sci.multivariate_normal(mean=d.mean().values.T, cov=d.cov().values)
x, y = np.mgrid[0:40, 710:750]
pos = np.empty(x.shape + (2,))
##3次元目にxとyを代入
pos[:,:,0] = x
pos[:,:,1] = y
##x,y座標とそれに対応する密度を算出する
cont = plt.contour(x,y,irlidar.pdf(pos))
##等高線に値を書き込むためのフォーマット指定
cont.clabel(fmt='%1.1e')
plt.show()

#共分散行列の意味
##共分散行列にnp.array([0,20],[20,0])を足して楕円を傾けてみる
c = d.cov().values + np.array([[0,20],[20,0]])
tmp = sci.multivariate_normal(mean=d.mean().values.T, cov=c)
cont = plt.contour(x,y,tmp.pdf(pos))
cont.clabel(fmt='%1.1e')
plt.show()
#別のデータ(sensor_data_200.txt)で負の共分散について考察する
##データのインポート
print("sensor_data_200.txt")
data2 = pd.read_csv("sensor_data_200.txt", delimiter=" ",
                    header=None, names=("date","time","ir","lidar"))
d2 = data2.loc[:, ["ir", "lidar"]] #光センサとLiDARのデータだけ抽出する
##平均・共分散行列
print("平均")
mean2 = d2.mean()
print(mean2)
print("共分散行列")
cov2 = d2.cov()
print(cov2)
##描画
sns.jointplot(data = d2, x = "ir", y = "lidar", kind="kde", fill=True, thresh=0) #add "fill=True, thresh=0" for jointplot latest version
plt.show()
##モデル化描画
irlidar2 = sci.multivariate_normal(mean=mean2.values.T, cov=cov2.values)
x2, y2 = np.mgrid[280:340, 190:230]
pos2 = np.empty(x2.shape + (2,))
##3次元目にxとyを代入
pos2[:,:,0] = x2
pos2[:,:,1] = y2
##x,y座標とそれに対応する密度を算出する
cont = plt.contour(x2,y2,irlidar2.pdf(pos2))
##等高線に値を書き込むためのフォーマット指定
cont.clabel(fmt='%1.1e')
plt.show()
