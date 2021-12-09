# ！usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/12/8 14:20
# @Author : LucXiong
# @Project : Model
# @File : SCA_new.py

"""
Ref:https://github.com/luizaes/sca-algorithm
S. Mirjalili, SCA: A Sine Cosine Algorithm for Solving Optimization Problems, Knowledge-based Systems, in press, 2015, DOI: http://dx.doi.org/10.1016/j.knosys.2015.12.022
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from test_func import *

class sca():
    def __init__(self, pop_size=5, n_dim=2, a=2, lb=-1e5, ub=1e5, max_iter=20, func=None):
        self.pop = pop_size
        self.n_dim = n_dim
        self.a = a # 感知概率
        self.func = func
        self.max_iter = max_iter  # max iter

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))] # y = f(x) for all particles

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

    def update(self, i):
        r1 = self.a - i * ((self.a) / self.max_iter)
        for j in range(self.pop):
            for k in range(self.n_dim):
                r2 = 2 * math.pi * random.uniform(0.0, 1.0)
                r3 = 2 * random.uniform(0.0, 1.0)
                r4 = random.uniform(0.0, 1.0)
                if r4 < 0.5:
                    try:
                        self.X[j][k] = self.X[j][k] + (r1 * math.sin(r2) * abs(r3 * self.gbest_x[k] - self.X[j][k]))
                    except:
                        self.X[j][k] = self.X[j][k] + (r1 * math.sin(r2) * abs(r3 * self.gbest_x[0][k] - self.X[j][k]))
                else:
                    try:
                        self.X[j][k] = self.X[j][k] + (r1 * math.cos(r2) * abs(r3 * self.gbest_x[k] - self.X[j][k]))
                    except:
                        self.X[j][k] = self.X[j][k] + (r1 * math.cos(r2) * abs(r3 * self.gbest_x[0][k] - self.X[j][k]))
        self.X = np.clip(self.X, self.lb, self.ub)
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # Function for fitness evaluation of new solutions


    def run(self):
        for i in range(self.max_iter):
            self.update(i)
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

if __name__ == '__main__':
    n_dim = 2
    lb = [-512 for i in range(n_dim)]
    ub = [512 for i in range(n_dim)]
    demo_func = f23
    sca = sca(n_dim=2, pop_size=40, max_iter=150, lb=lb, ub=ub, func=demo_func)
    sca.run()
    print('best_x is ', sca.gbest_x, 'best_y is', sca.gbest_y)
    # f22 best_x is  [-0.08631097  0.71930416] best_y is -1.0311910818775485
    # f23 best_x is  [449.26653876 465.54725104] best_y is -937.0562555095239
    print(f'{demo_func(sca.gbest_x)}\t{sca.gbest_x}')
    plt.plot(sca.gbest_y_hist)
    plt.show()
