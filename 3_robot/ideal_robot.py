#! /usr/bin/env python3

#詳解確率ロボティクス第3章
import matplotlib
matplotlib.use('TkAgg') #not using nbagg because it is for jupyternotebook
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
        self.debug = debug  #Flag for debug
        self.time_span = time_span
        self.time_interval = time_interval

    def append(self, obj):
        self.objects.append(obj)

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
    def __init__(self, pose, agent=None, sensor=None, color="black"):
        self.pose = pose  #Robot's Position and angle = State
        self.r = 0.2  #Robot's Radius
        self.color = color  #Robot's color.
        self.agent = agent
        self.poses = [pose] #For drwaing trajectory
        self.sensor = sensor
        
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
        if self.sensor and len(self.poses)>1:
            self.sensor.draw(ax, elems, self.poses[-2])

    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        

class Agent:
    def __init__(self, nu = 0, omega = 0):
        self.nu = nu
        self.omega = omega

    def decision(self, observation = None):
        return self.nu, self.omega

class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None

    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmark", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))

class Map:
    def __init__(self):
        self.landmarks = []

    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)

    def draw(self, ax, elems):
        for im in self.landmarks:
            im.draw(ax, elems)


##Class for Observation
class IdealCamera:
    def __init__(self, env_map,
                 distance_range=(0.5,100),
                 direction_range=(-math.pi/3, math.pi/3)):
        self.map = env_map
        self.lastdata = []
        self.distance_range = distance_range
        self.direction_range = direction_range

    def visible(self, polarpos):
        if polarpos is None:
            return False

        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1] and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]

    def data(self, cam_pose):
        observed = []
        for im in self.map.landmarks:
            p = self.observation_function(cam_pose, im.pos)
            if self.visible(p):
                 observed.append((p, im.id))
        self.lastdata = observed
        return observed

    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos -cam_pose[0:2]
        phi = math.atan2(diff[1],diff[0]) - cam_pose[2]
        while phi >= np.pi:
            phi -= 2*np.pi
        while phi < np.pi:
            phi += 2*np.pi
        return np.array([np.hypot(*diff), phi]).T

    def draw(self, ax, elems, cam_pose):
        for im in self.lastdata:
            x,y,theta = cam_pose
            distance, direction = im[0] #fixed by ChatGPT
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x,lx],[y,ly],color="pink")
            
# 世界座標系の描画
if __name__ == '__main__':
    world = World(10, 0.1)  # 世界オブジェクトを生成
    m = Map()
    m.append_landmark(Landmark(2,-2))
    m.append_landmark(Landmark(-1,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)

    straight = Agent(0.2, 0.0)
    circling = Agent(0.2, 19.9/180*math.pi) #speed: 0.2mps, angle_speed: 10 dps
    robot1 = IdealRobot(np.array([2, 3, pi/6]).T, sensor=IdealCamera(m), agent=straight, color="blue")
    robot2 = IdealRobot(np.array([-2, -1, pi/6 * 5]).T, sensor=IdealCamera(m), agent=circling, color="red")
    world.append(robot1)
    world.append(robot2)
    world.draw()
    result = subprocess.run(['python3', './ideal_robot_viewer.py'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error:", result.stderr)
