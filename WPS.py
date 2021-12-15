# ！usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/12/7 14:45
# @Author : LucXiong
# @Project : Model
# @File : wps.py

"""
There are 2 kinds of wolf pack search algorithm.
The first one called wolf pack search(wps) proposed by Chenguang Yang, Xuyan Tu and Jie Chen in thier paper "Algorithm of Marriage in Honey Bees Optimization Based on the Wolf Pack Search" at 2007 International Conference on Intelligent Pervasive Computing in 2007. The wps code could ref: https://github.com/AlexanderKlanovets/swarm_algorithms/commit/78834820cadbcadb6902e3c171a2a8581255c542
The second one called wolf pack algorithm proposed by 吴虎胜,张凤鸣,吴庐山 in their paper "一种新的群体智能算法-狼群算法" at Systems Engineering and Electronics in 2013.
In fact, they are 2 different kinds of algorithm. WPS was shown in  this file, and WPA was in wpa.py.(I meet some problems in wpa.py, so it haven't been upload.)
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import test_function

class wps():
    def __init__(self, n_dim=2, pop_size=50, max_iter=150, lb=-1e5, ub=1e5, step=0.5, func=None):
        # D - dimension of the search space;
        # N - number of wolves to generate;
        # step, l_near, t_max, beta - model parameters;
        # fitness_function - function to optimize;
        # w_range - range of particles' coordinates' values (from -range to range);
        # iter_num - maximum number of iterations.
        # (self, n_dim, pop_size, step, fitness_function, w_range, iter_num, l_near=0, t_max=0, beta=0):
        self.n_dim = n_dim
        self.pop = pop_size
        self.step = step # 影响朝最佳点靠近的速度
        self.func = func
        self.max_iter = max_iter

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles

        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = [np.inf for i in range(self.pop)]  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        for i in range(len(self.Y)):
            if self.pbest_y[i] > self.Y[i]:
                self.pbest_x[i] = self.X[i]
                self.pbest_y[i] = self.Y[i]

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
        # 参考樽海鞘群算法
        # self.step = 2 * math.exp(-(4 * ((i + 1) / self.max_iter)) ** 2)
        for i in range(self.pop):
            if all(self.X[i] != self.gbest_x[0]): # self.Y[i] != self.gbest_y
                try:
                    self.X[i] += self.step * (self.gbest_x[0] - self.X[i]) / np.linalg.norm(self.gbest_x[0] - self.X[i])
                except:
                    self.X[i] += self.step * (self.gbest_x - self.X[i]) / np.linalg.norm(self.gbest_x - self.X[i])
        self.X = np.clip(self.X, self.lb, self.ub)
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # Function for fitness evaluation of new solutions

    def run(self):
        for iter in range(self.max_iter):
            self.update() # self.step可以选择参考樽海鞘群，随迭代次数发生改变
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

if __name__ == '__main__':
    n_dim = 2
    lb = [-5.12 for i in range(n_dim)]
    ub = [5.12 for i in range(n_dim)]
    demo_func = test_function.f22
    ssa = wps(pop_size=50, n_dim=n_dim, max_iter=300, lb=lb, ub=ub, func=demo_func, step=0.5)
    ssa.run()
    print('best_x is ', ssa.gbest_x, 'best_y is', ssa.gbest_y)
    print(f'{demo_func(ssa.gbest_x)}\t{ssa.gbest_x}')
    plt.plot(ssa.gbest_y_hist)
    plt.show()
