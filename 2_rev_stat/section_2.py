#!/usr/bin/env python3

#詳解確率ロボティクス 第2章（確率統計の復習）

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

#1. 前処理
##データの読み取り
data = pd.read_csv("sensor_data_200.txt", delimiter=" ",
                   header = None, names = ("data","time","ir","lidar"))
##データの先頭の表示
print("data.head()")
print(data.head())
print("--------")
print("data[\"lidar\"][0:5]")
print(data["lidar"][0:5])


#2. ヒストグラムの描画
data["lidar"].hist(bins=max(data["lidar"])-min(data["lidar"]), align="left")
plt.show()


#3. 平均
##センサの平均値の計算
print("--------")
print("平均値の計算")
mean1 = sum(data["lidar"].values)/len(data["lidar"].values) #.valuesはheaderを除いて値だけの配列にしている．
mean2 = data["lidar"].mean()
print("定義通りの計算",mean1,", mean()関数による計算",mean2)

##ヒストグラムの上に平均値を重ねて描画する
data["lidar"].hist(bins=max(data["lidar"])-min(data["lidar"]), color="orange", align="left")
plt.vlines(mean2, ymin=0, ymax=5000, color="red") #高さを指定すれば縦の線をvlineで描ける!
plt.show()


#4. 標本分散と不偏分散
##定義から計算
print("--------")
print("定義から計算する")
zs = data["lidar"].values#()は要らない！
n = len(zs)
mean = sum(zs)/n
diff_square = [(z - mean)**2 for z in zs]

sampling_var = sum(diff_square)/n
unbiased_var = sum(diff_square)/(n-1)
print("標本分散: ", sampling_var)
print("不偏分散: ", unbiased_var)

##Pandasの関数を用いて計算
print("Pandasを使用")
pandas_sampling_var = zs.var(ddof=False) #標本分散
pandas_default_var = zs.var() #デフォルト（不偏分散）
print("標本分散: ", pandas_sampling_var)
print("不偏分散: ", pandas_default_var)

##NumPyの関数を用いて計算
print("Numpyを使用")
numpy_default_var = np.var(zs)
numpy_unbiased_var = np.var(zs, ddof=1) #不偏分散(ddofはデータの自由度)
print("標本分散: ", numpy_default_var)
print("不偏分散: ", numpy_unbiased_var)


#5. 標準偏差を計算
print("標準偏差を計算")
stddev1 = math.sqrt(sampling_var)
stddev2 = math.sqrt(unbiased_var)
pandas_stddev = zs.std()
print("math.sqrt(sampling_var)",stddev1)
print("math.sqrt(unbiased_var)",stddev2)
print("math.sqrt",pandas_stddev)


#6. 確率分布
print("------")
print("確率分布")
##頻度をデータフレームにする
freqs = pd.DataFrame(data["lidar"].value_counts())
print(freqs.transpose()) #横向きに出力する
##確率の列を追加する
freqs["probs"] = freqs/len(data["lidar"]) #freqs["lidar"]/len(data["lidar"])はダメだった
print(freqs.transpose()) #横向きに出力する
##確率の和が1になることの確認
print("確率の和は",sum(freqs["probs"]))

##freqsの描画(度数分布表を並べ替える)
freqs["probs"].sort_index().plot.bar()
plt.show()


#7. 確率分布を用いたシミュレーション
def drawing(): #関数としてサブルーチン化
    return freqs.sample(n=1, weights="probs").index[0] #1個とりだす．probの確率に応じてとる． index[0]でセンサの値を取得する
ret = drawing()
print("ドローの結果は",ret)

#2.3 確率モデル
#ガウス分布の当てはめ
