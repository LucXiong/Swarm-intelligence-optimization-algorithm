# ！usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/12/15 15:57
# @Author : LucXiong
# @Project : Model
# @File : PSO.py

"""
Ref:https://github.com/guofei9987/scikit-opt
已经有了很多可以调用粒子群算法的库，但是在高维情形下的寻优效果很差，所以决定自己再写一个
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import test_function

class PSO():
    def __init__(self, func, n_dim=None, pop=40, max_iter=150, lb=-1e5, ub=1e5, w=0.8, c1=0.5, c2=0.5):
        self.func = func
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.n_dim))  # speed of particles
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

    def update_V(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

    def update(self):
        for i in range(self.pop):
            self.X[i] += self.V[i]
        self.X = np.clip(self.X, self.lb, self.ub)
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]

    def run(self):
        for iter_no in range(self.max_iter):
            self.update_V()
            self.update()
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

if __name__ == '__main__':
    n_dim = 30
    lb = [-100 for i in range(n_dim)]
    ub = [100 for i in range(n_dim)]
    demo_func = test_function.fu2
    pop_size = 100
    max_iter = 1000
    pso = PSO(func=demo_func, n_dim=n_dim, pop=100, max_iter=1000, lb=lb, ub=ub, w=0.8, c1=0.5, c2=0.5)
    best_x, bext_y = pso.run()
    print(f'{demo_func(pso.gbest_x)}\t{pso.gbest_x}')
    # plt.plot(pso.gbest_y_hist)
    # plt.show()