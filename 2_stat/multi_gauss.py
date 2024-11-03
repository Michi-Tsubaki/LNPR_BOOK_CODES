#!/usr/bin/env python3

#確率ロボティクス第2章　確率統計の復習
##「多次元ガウス分布」

#ライブラリのインポート
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("sensor_data_700.txt", delimiter=" ",
                   header=None, names=("data","time","ir","lidar"))

d = data[(data["time"]<160000) & (data["time"] >= 120000)] #12時から16時までのデータを抽出
d = d.loc[:,["ir","lidar"]] #LiDARの値と光センサの値

sns.jointplot(x="ir", y="lidar", data=d, kind="kde")
plt.show()

#平均・共分散行列
print("平均")
mean = d.mean()
print(mean)
print("共分散行列")
cov = d.cov()
print(cov)
