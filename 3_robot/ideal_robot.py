#! /usr/bin/env python3

#詳解確率ロボティクス第3章
import matplotlib
matplotlib.use('nbagg')  # Jupyter Notebookでのアニメーション表示用
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import subprocess

pi = math.pi 

# 世界座標系のクラスを定義する
class World:
    def __init__(self, debug=False):
        self.objects = []  # ロボットのオブジェクトを格納するリスト
        self.debug = debug  # デバッグ用のフラグ

    def append(self, obj):
        self.objects.append(obj)  # オブジェクトをリストに追加

    def draw(self):
        fig = plt.figure(figsize=(8, 8))  # 図のサイズを設定
        ax = fig.add_subplot(111)  # 1つのサブプロットを作成
        ax.set_aspect('equal')  # アスペクト比を等しく設定
        ax.set_xlim(-5, 5)  # X軸の範囲を設定
        ax.set_ylim(-5, 5)  # Y軸の範囲を設定
        ax.set_xlabel("X", fontsize=20)  # X軸のラベル
        ax.set_ylabel("Y", fontsize=20)  # Y軸のラベル

        elems = []  # 描画要素を格納するリスト
        if self.debug:
            for i in range(1000):
                self.one_step(i, elems, ax)  # デバッグ時はアニメーションを表示
        else:
            # アニメーションを設定し、動画として保存
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax), frames=10, interval=1000, repeat=False)
            #self.ani.save('animation.mp4', writer='ffmpeg')  # 動画を保存
            self.ani.save('animation.gif', writer='imagemagick')  # GIF形式で保存
            plt.show()  # プロットを表示

    def one_step(self, i, elems, ax):
        # 描画要素をクリア
        while elems:
            elems.pop().remove()
        # 現在の時刻を表示
        elems.append(ax.text(-4.4, 4.5, "t=" + str(i), fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)  # 各オブジェクトを描画

# ロボットのクラスを定義する
class IdealRobot:
    def __init__(self, pose, color="black"):
        self.pose = pose  # ロボットの位置と向き
        self.r = 0.2  # ロボットの半径
        self.color = color  # ロボットの色

    def draw(self, ax, elems):
        # ロボットの位置と向きを取得
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)  # 向きに応じたX座標
        yn = y + self.r * math.sin(theta)  # 向きに応じたY座標
        elems.append(ax.plot([x, xn], [y, yn], color=self.color)[0])  # 向きを示す線を描画
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)  # ロボットの円を作成
        elems.append(ax.add_patch(c))  # 円をプロットに追加

# 世界座標系の描画
world = World()  # 世界オブジェクトを生成
robot1 = IdealRobot(np.array([2, 3, pi/6]).T)  # ロボット1を生成
robot2 = IdealRobot(np.array([-2, -1, pi/6 * 5]).T, "red")  # ロボット2を生成
world.append(robot1)  # 世界にロボット1を追加
world.append(robot2)  # 世界にロボット2を追加
world.draw()  # 世界を描画
subprocess.run(['python3', './viewer.py'])
