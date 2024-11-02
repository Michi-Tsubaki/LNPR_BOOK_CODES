#! /usr/bin/env python3

#詳解確率ロボティクス第3章
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import subprocess

pi = math.pi 

#Define World Coordination.
class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []  # ロボットのオブジェクトを格納するリスト
        self.debug = debug  # デバッグ用のフラグ
        self.time_span = time_span
        self.time_interval = time_interval

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
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                         frames=int(self.time_span/self.time_interval)+1,
                                         interval=int(self.time_interval*1000), repeat=False)
            self.ani.save('ideal_robot_ani.gif', writer='pillow')  # Use Pillow instead
            #plt.show()  # プロットを表示

    def one_step(self, i, elems, ax):
        while elems: elems.pop().remove() #clear all
        elems.append(ax.text(-4.4, 4.5, "t=" + str(i), fontsize=10)) #put current time stamp
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(1.0)

#Define Robot Class
class IdealRobot:
    def __init__(self, pose, agent=None, color="black"):
        self.pose = pose  # ロボットの位置と向き
        self.r = 0.2  # ロボットの半径
        self.color = color  #Robot's color.
        self.agent = agent
        self.poses = [pose] #For drwaing trajectory
        
    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2] #theta
        if math.fabs(omega) < 1e-10: ##where omega is very close to zero.
            return pose + np.array([nu*math.cos(t0), nu*math.sin(t0), omega]) *time
        else:
            return pose + np.array([nu/omega*(math.sin(t0+omega*time)-math.sin(t0)), nu/omega*(-math.cos(t0+omega*time)+math.cos(t0)), omega*time])

    def draw(self, ax, elems):
        x, y, theta = self.pose #get position from self. 
        xn = x + self.r * math.cos(theta)  # 向きに応じたX座標
        yn = y + self.r * math.sin(theta)  # 向きに応じたY座標
        elems.append(ax.plot([x, xn], [y, yn], color=self.color)[0])  # 向きを示す線を描画
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)  # ロボットの円を作成
        elems.append(ax.add_patch(c))  # 円をプロットに追加
        self.poses.append(self.pose)  #For drawing trajectory
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")

    def one_step(self, time_interval):
        if not self.agent: return
        nu, omega = self.agent.decision()
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        

class Agent:
    def __init__(self, nu = 0, omega = 0):
        self.nu = nu
        self.omega = omega

    def decision(self, observation = None):
        return self.nu, self.omega

# 世界座標系の描画
world = World(10, 1)  # 世界オブジェクトを生成
straight = Agent(0.2, 0.0)
circling = Agent(0.2, 19.9/180*math.pi) #speed: 0.2mps, angle_speed: 10 dps
robot1 = IdealRobot(np.array([2, 3, pi/6]).T, straight)  # ロボット1を生成
robot2 = IdealRobot(np.array([-2, -1, pi/6 * 5]).T, circling,"red")  # ロボット2を生成
robot3 = IdealRobot(np.array([0, 0, 0]).T, Agent(), "blue")
world.append(robot1)
world.append(robot2)
world.append(robot3)
world.draw()

result = subprocess.run(['python3', './ideal_robot_viewer.py'], capture_output=True, text=True)
if result.returncode != 0:
    print("Error:", result.stderr)
