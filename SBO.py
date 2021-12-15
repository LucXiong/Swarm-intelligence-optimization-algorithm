# ！usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/12/3 17:36
# @Author : LucXiong
# @Project : Model
# @File : SBO.py

"""
Ref:https://blog.csdn.net/u011835903/article/details/107857884
"""

import random  # random Function
import numpy as np # numpy operations
import matplotlib.pyplot as plt
import math
import test_function

class SBO():
    def __init__(self, pop_size=50, n_dim=2, alpha=0.94, lb=-1e5, ub=1e5, max_iter=1000, func=None):
        self.pop = pop_size
        self.n_dim = n_dim
        self.alpha = alpha # 步长的最大阈值
        self.func = func
        self.max_iter = max_iter  # max iter
        self.z = 0.02 # z是缩放比例因子
        self.r_mutate = 0.05  # 变异概率

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))] # y = f(x) for all particles

        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = [np.inf for i in range(self.pop)]  # best image of every particle in history
        self.fit = [1 / (1 + self.Y[i]) if self.Y[i] > 0 else 1 - self.Y[i] for i in range(self.pop)]
        self.prob = [self.fit[i] / sum(self.fit) for i in range(self.pop)]
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

    def cal_prob(self):
        self.fit = [1 / (1 + self.Y[i]) if self.Y[i] > 0 else 1 - self.Y[i] for i in range(self.pop)]
        self.prob = [self.fit[i] / sum(self.fit) for i in range(self.pop)]

    def update(self, iter_num):
        idx_min = self.Y.index(min(self.Y))
        for i in range(self.pop):
            # roulette wheel
            for k in range(self.n_dim):
                select_list = []
                while len(select_list) < 1:
                    select_list = []
                    r = np.random.rand(1)
                    for kk in range(len(self.prob)):
                        if self.prob[kk] > (r[0]):
                            select_list.append(kk)
                j = random.choice(select_list)
                lemta = self.alpha / (1 + self.prob[j])
                try:
                    self.X[i, k] += lemta * 0.5 * (self.X[j, k] + self.gbest_x[0][k] - 2 * self.X[i, k])
                except:
                    self.X[i, k] += lemta * 0.5 * (self.X[j, k] + self.gbest_x[k] - 2 * self.X[i, k])

            # 变异
            if np.random.rand(1) < self.r_mutate and i != idx_min:
                for j in range(self.n_dim):
                    # 正态分布 Satin bowerbird optimizer: A new optimization algorithm to optimize
                    # ANFIS for software development effort estimation.pdf
                    self.X[i][j] += self.z * np.random.normal() * (self.ub[j] - self.lb[j])
        self.X = np.clip(self.X, self.lb, self.ub)
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]

    # def mutate(self):
    #     # 没有用上这个函数
    #     idx_min = self.Y.index(min(self.Y))
    #     for i in range(self.pop):
    #         if np.random.rand(1)[0] < self.r_mutate and i != idx_min:
    #             for j in range(self.n_dim):
    #                 self.X[i][j] += self.z * np.random.normal() * (self.ub[j] - self.lb[j])
    #     self.X = np.clip(self.X, self.lb, self.ub)
    #     self.Y = [self.func(self.X[i]) for i in range(len(self.X))]

    def run(self):
        for i in range(self.max_iter):
            print(i)
            self.update(i)
            self.cal_prob()
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

if __name__ == '__main__':
    # 寻优效果不错，但是计算时间较长，有一部分原因是我频繁整体计算适应度值
    n_dim = 30
    lb = [-100 for i in range(n_dim)]
    ub = [100 for i in range(n_dim)]
    demo_func = test_function.fu5
    pop_size = 100
    max_iter = 100
    sbo = SBO(n_dim=n_dim, pop_size=pop_size, max_iter=max_iter, lb=lb, ub=ub, func=demo_func)
    best_x, bext_y = sbo.run()
    print(f'{demo_func(sbo.gbest_x)}\t{sbo.gbest_x}')
    plt.plot(sbo.gbest_y_hist)
    plt.show()

