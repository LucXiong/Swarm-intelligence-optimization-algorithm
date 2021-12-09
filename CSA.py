# ！usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/12/3 15:23
# @Author : LucXiong
# @Project : Model
# @File : CSA.py

"""
Ref:
[1] https://github.com/Luan-Michel/CrowSearchAlgorithmPython/blob/master
# /CrowSearchAlgorithm.py
[2] Alireza Askarzadeh, Anovel metaheuristic method for solving constrained
# engineering optimization problems: Crow search algorithm, Computers &
# Structures, Vol. 169, 1-12, 2016.
"""

import random  # random Function
import numpy as np # numpy operations
import math  # ceil function

class CSA():
    def __init__(self, pop_size=5, n_dim=2, ap=0.1, lb=-1e5, ub=1e5, max_iter=20, func=None):
        self.pop = pop_size
        self.n_dim = n_dim
        self.ap = ap # 感知概率
        self.func = func
        self.max_iter = max_iter  # max iter
        self.fly_length = [2, 2] # 飞行距离，可以考虑是否采用莱维飞行或者随迭代次数改变
        # 或许也和变量维数相关
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

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        print(self.Y)
        return self.Y

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
        num = np.array([random.randint(0, self.pop - 1) for _ in range(self.pop)])  # Generation of random candidate crows for following (chasing)
        xnew = np.empty((self.pop, self.n_dim))
        for i in range(self.pop):
            if (random.random() > self.ap):
                for j in range(self.n_dim):
                    self.X[(i, j)] = self.X[(i, j)] + self.fly_length[j] * ((random.random()) * (self.pbest_x[(num[i], j)] - self.X[(i, j)]))
            else:
                for j in range(self.n_dim): # 随机生成或许可以考虑采用莱维飞行
                    self.X[(i, j)] = self.lb[j] - (self.lb[j] - self.ub[j]) * random.random()
        self.X = np.clip(self.X, self.lb, self.ub)
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # Function for fitness evaluation of new solutions

    def run(self):
        for iter in range(self.max_iter):
            self.update()
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y


def demo_func(x):
    x1 = x[0]
    x2 = x[1]
    result = 4 * x1 ** 2 - 2.1 * x1 ** 4 + x1 ** 6 /3 + x1 * x2 - 4 * x2 ** 2 + 4 * x2 ** 4
    return result

if __name__ == '__main__':
    lb = [-5, -5]
    ub = [5, 5]
    csa = CSA(n_dim=2, pop_size=40, max_iter=150, lb=lb, ub=ub, func=demo_func)
    best_x, bext_y = csa.run()
    print(csa.X)
    print(csa.Y)
    print('sss')
    print(best_x)
    print(bext_y)
    print(demo_func(best_x))

