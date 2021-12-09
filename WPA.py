# ！usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/12/7 14:06
# @Author : LucXiong
# @Project : Model
# @File : wpa.py

"""
There are 2 kinds of wolf pack search algorithm.
The first one called wolf pack search(wps) proposed by Chenguang Yang, Xuyan Tu and Jie Chen in thier paper "Algorithm of Marriage in Honey Bees Optimization Based on the Wolf Pack Search" at 2007 International Conference on Intelligent Pervasive Computing in 2007. The wps code could ref: https://github.com/AlexanderKlanovets/swarm_algorithms/commit/78834820cadbcadb6902e3c171a2a8581255c542
The second one called wolf pack algorithm proposed by 吴虎胜,张凤鸣,吴庐山 in their paper "一种新的群体智能算法-狼群算法" at Systems Engineering and Electronics in 2013.
In fact, they are 2 different kinds of algorithm. WPA was shown in  this file, and WPS was in wps.py.

Ref:git@github.com:xzltc/WPA.git
Ref:https://blog.csdn.net/weixin_37978667/article/details/85112709
>吴虎胜,张凤鸣,吴庐山.一种新的群体智能算法——狼群算法[J].系统工程与电子技术, 2013, 35(11): 2430-2438.
>周强,周永权-种基于领导者策略的狼群搜索算法[J.计算机应用研究,2013,30(9):2629-2632.
"""
import numpy as np
import math
import random
import heapq
from test_func import *
import matplotlib.pyplot as plt


class wpa():
    def __init__(self, pop_size=50, n_dim=2, alpha=4, beta=6, w=100, lb=-1e5, ub=1e5, max_iter=300, func=None):
        self.pop = pop_size
        self.n_dim = n_dim
        self.func = func
        self.max_iter = max_iter  # max iter
        self.alpha = alpha  # 探狼比例因子 （取[n／(α+1)，n／α]之间的整数）
        self.beta = beta  # 狼群更新比例因子
        self.w = w  # 距离判定因子
        self.S = 1000  # 步长因子
        self.number_T = np.random.randint(self.pop / (self.alpha + 1), self.pop / self.alpha)  # 探狼数量
        self.h = 10 # 探狼探寻方向总数
        self.T_max = 30  # 最大游走次数

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.step_a = (self.ub - self.lb) / self.S  # 探狼游走步长
        self.step_b = 2 * self.step_a  # 猛狼奔袭步长
        self.step_c = self.step_a / 2  # 围攻步长

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles

        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = [self.Y[i] for i in range(self.pop)]  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        # self.need_update = self.pbest_y > self.Y
        # print(self.need_update)
        # print(self.pbest_y)
        # print(self.Y)
        for i in range(len(self.Y)):
            if self.pbest_y[i] > self.Y[i]:
                self.pbest_x[i] = self.X[i]
                self.pbest_y[i] = self.Y[i]
        # self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        # self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)
        # print(self.pbest_x)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.index(min(self.pbest_y))
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def update(self):
        L_Wolf_Y = np.min(self.Y)  # 记录头狼的值
        L_Wolf_index = np.argmin(self.Y)  # 头狼在狼群(wol)f_colony_X)中下标位置
        L_Wolf_X = self.X[L_Wolf_index]  # 记录头狼的变量信息
        L_Wolf = [L_Wolf_Y, L_Wolf_index]

        # 初始化探狼
        T_Wolf_V = heapq.nsmallest(self.number_T, self.Y.copy())
        T_Wolf = self.find_index(T_Wolf_V, self.Y)  # 0:探狼在狼群中位置 1:解
        # 初始化猛狼
        M_Wolf_V = heapq.nlargest(self.pop - self.number_T - 1, self.Y.copy())
        M_Wolf = self.find_index(M_Wolf_V, self.Y)  # 0:探狼在狼群中位置 1:解
        self.detective_wolf(T_Wolf, L_Wolf_Y, L_Wolf_X, L_Wolf)
        # 需要修改，参考一种新的群体智能算法 狼群算法> 2013.11 式（3）
        # 言w增大会加速算法收敛，但W过大会使得人工狼很难进入围攻行为，缺乏对猎物的精细搜索。
        d_near = 0
        for d in range(self.n_dim):
            d_near += ((1 / (1 * self.w * self.n_dim)) * (self.ub[d] - self.lb[d]))  # dnear值，判定距离
        self.call_followers(d_near, M_Wolf, L_Wolf_Y, L_Wolf_X, L_Wolf_index)
        self.siege(M_Wolf, L_Wolf_X, L_Wolf_Y, L_Wolf_index)
        self.live()

    def find_index(self, value, wolf_colony):
        ret = np.zeros((len(value), 2))

        for i in range(0, len(value)):
            ret[i][0] = wolf_colony.index(value[i])
            ret[i][1] = value[i]
        return ret

    def detective_wolf(self, T_Wolf, L_Wolf_Y, L_Wolf_X, L_Wolf):
        # 探狼开始游走
        for i in range(0, T_Wolf.shape[0]):
            H = np.random.randint(2, self.h)  # 尝试的方向
            single_T_Wolf = self.X[int(T_Wolf[i][0])]  # 当前探狼
            optimum_value = T_Wolf[i][1]  # 游走的最优解
            optimum_position = single_T_Wolf.copy()  # 游走最优位置
            find = False

            # 探狼游走行为 一旦探狼发现目标 -> 开始召唤
            for t in range(0, self.T_max):
                # 在初始位置朝H个方向试探，直到找到一个优解
                for p in range(1, H + 1):
                    single_T_Wolf_trial = single_T_Wolf.copy()
                    # 一种新的群体智能算法 狼群算法> 2013.11 式（1）
                    # h越大探狼搜寻得越精细但同时速度也相对较慢
                    # 这个部分有点像果蝇的搜素
                    single_T_Wolf_trial = single_T_Wolf_trial + math.sin(2 * math.pi * p / H) * self.step_a # 根据原式不需要转化类型为int
                    # 一种基于领导者策略的狼群搜索算法> 2013.9 式（2）
                    # single_T_Wolf_trial = single_T_Wolf_trial + np.random.uniform(-1, 1,(single_T_Wolf_trial.shape[0],)) * self.step_a

                    single_T_Wolf_V = self.func(single_T_Wolf_trial) # [0]

                    # 探狼转变为头狼
                    if L_Wolf_Y > single_T_Wolf_V:
                        find = True
                        L_Wolf_Y = single_T_Wolf_V  # 更新头狼解
                        L_Wolf_X = single_T_Wolf_trial
                        L_Wolf_index = int(T_Wolf[i][0])  # 更新头狼下标
                        self.X[L_Wolf_index] = single_T_Wolf_trial  # 更新头狼位置参数

                        T_Wolf = np.delete(T_Wolf, i, axis=0)  # 探狼转为头狼，发起召唤，删除探狼
                        break

                    elif optimum_value > single_T_Wolf_V:
                        optimum_value = single_T_Wolf_V
                        optimum_position = single_T_Wolf_trial

                else:
                    # print(" > 第%d只探狼完成第%d次游走,未发现猎物" % ((i + 1), (t + 1))) # 记录上次的最优位置
                    single_T_Wolf = optimum_position

                if find is True:
                    break

            if find is True:
                print("第%d只探狼发现猎物 %f" % (i, single_T_Wolf_V))
                break

            else:
                # 若游走完成探狼没找到猎物，更新所有游走过程中最优的一次位置
                self.X[int(T_Wolf[i][0])] = optimum_position

    def call_followers(self, d_near, M_Wolf, L_Wolf_Y, L_Wolf_X, L_Wolf_index):
        # 召唤行为
        surrounded = False
        # 所有猛狼进入围攻范围才能结束
        while ~surrounded:
            ready = 0
            for m in range(0, M_Wolf.shape[0]):
                s_m_index = int(M_Wolf[m][0])  # 猛狼在狼群中下标
                single_M_Wolf = self.X[s_m_index]  # 猛狼变量
                d = np.abs(L_Wolf_X - single_M_Wolf)
                dd = np.sum(d)

                while d_near < dd:
                    for d in range(self.n_dim):
                        single_M_Wolf[d] = single_M_Wolf[d] + self.step_b[d] * (L_Wolf_X[d] - single_M_Wolf[d]) / np.abs(L_Wolf_X[d] - single_M_Wolf[d])
                    single_M_Wolf_V = self.func(single_M_Wolf)

                    # 更新猛狼位置
                    self.X[s_m_index] = single_M_Wolf
                    M_Wolf[m][1] = single_M_Wolf_V

                    d = np.abs(L_Wolf_X - single_M_Wolf)
                    dd = np.sum(d)

                    if L_Wolf_Y > single_M_Wolf_V:
                        # 头狼变猛狼
                        M_Wolf[m][0] = L_Wolf_index
                        M_Wolf[m][1] = L_Wolf_Y
                        self.X[L_Wolf_index] = L_Wolf_X

                        # 猛狼变头狼
                        L_Wolf_Y = single_M_Wolf_V
                        L_Wolf_X = single_M_Wolf
                        L_Wolf_index = s_m_index
                        self.X[L_Wolf_index] = single_M_Wolf
                        break

                if d_near > dd:
                    ready += 1  # 围攻就绪态+1
                else:
                    break

            # 所有猛狼是否就位
            if ready == M_Wolf.shape[0]:
                print(" > 所有猛狼已进入围攻状态")
                break

    def siege(self, M_Wolf, L_Wolf_X, L_Wolf_Y, L_Wolf_index):
        for m in range(0, M_Wolf.shape[0]):
            s_m_index = int(M_Wolf[m][0])
            single_M_Wolf = self.X[s_m_index]
            # 发起围攻，计算围攻后位置
            single_M_Wolf = single_M_Wolf + np.random.uniform(-1, 1) * self.step_c * np.abs(L_Wolf_X - single_M_Wolf)
            single_M_Wolf_V = self.func(single_M_Wolf)

            if L_Wolf_Y > single_M_Wolf_V:
                print(" > 发起围攻!目标更新   原值:%f => 现值:%f" % (L_Wolf_Y, single_M_Wolf_V))
                # 头狼变猛狼
                M_Wolf[m][0] = L_Wolf_index
                M_Wolf[m][1] = L_Wolf_Y
                self.X[L_Wolf_index] = L_Wolf_X

                # 猛狼变头狼
                L_Wolf_Y = single_M_Wolf_V
                L_Wolf_X = single_M_Wolf
                L_Wolf_index = s_m_index
                self.X[L_Wolf_index] = single_M_Wolf
        if self.gbest_y > single_M_Wolf_V:
            self.gbest_y = single_M_Wolf_V
            self.gbest_x = single_M_Wolf
            self.gbest_y_hist.append(self.gbest_y)
        else:
            self.gbest_y_hist.append(self.gbest_y)


    def live(self):
        # 强者生存行为
        wolf_colony_V = [self.func(self.X[i]) for i in range(len(self.X))]  # 重新计算现在所有狼狩猎的状态
        eliminate_number = np.random.randint(self.pop / (2 * self.beta), self.pop / self.beta)
        Bad_Wolf_V = heapq.nlargest(eliminate_number, wolf_colony_V.copy())
        Bad_Wolf = self.find_index(Bad_Wolf_V, wolf_colony_V)
        # 生成新的人工狼
        new_wolf = np.random.uniform(self.lb, self.ub, (eliminate_number, self.n_dim))
        # 淘汰最弱的狼
        for n in range(0, len(Bad_Wolf_V)):
            self.X[int(Bad_Wolf[n][0])] = new_wolf[n]

    def run(self):
        for i in range(self.max_iter):
            print(i)
            self.update()
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

if __name__ == '__main__':
    n_dim = 2
    lb = [-5.12 for i in range(n_dim)]
    ub = [5.12 for i in range(n_dim)]
    demo_func = f22
    #  def __init__(self, pop_size=50, n_dim=2, alpha=4, beta=6, w=500, lb=-1e5, ub=1e5, max_iter=300, func=None):
    wpa = wpa(pop_size=50, n_dim=n_dim, max_iter=1000, lb=lb, ub=ub, func=demo_func, alpha=4, beta=6, w=500)
    wpa.run()
    print('best_x is ', wpa.gbest_x, 'best_y is', wpa.gbest_y)
    # f22 is best_x is  [ 0.09110638 -0.71978079] best_y is -1.0312114438653428
    # f23 is best_x is  [-448.40744098  403.16544094] best_y is -894.7260832993893 效果不行
    print(f'{demo_func(wpa.gbest_x)}\t{wpa.gbest_x}')
    plt.plot(wpa.gbest_y_hist)
    plt.show()
