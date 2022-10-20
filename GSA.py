# ！usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2022/10/19 20:32
# @Author : LucXiong
# @Project : Model
# @File : GSA.py

'''
Ref:https://github.com/ravexina/GSA/blob/master/GSA.py
Ref:Rashedi E., Nezamabadi-Pour H., Saryazdi S. GSA: A Gravitational Search Algorithm[J]. Information Sciences, 2009, 179(13): 2232-48.
'''

import numpy as np
import matplotlib.pyplot as plt
import math
import test_function

class GSA():
    def __init__(self, func, n_dim=None, pop=40, max_iter=150, lb=-1e5, ub=1e5, alpha=0.1, G=0.9):
        self.func = func
        self.alpha = alpha
        self.G = G
        self.pop = pop  # number of particles
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        v_high = (self.ub - self.lb) # 速度设置为区间长度的一半
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.n_dim))  # speed of particles
        self.Y = [self.func(self.X[i]) for i in range(self.pop)]  # y = f(x) for all particles
        self.q = [1 for i in range(self.pop)]
        self.M = [1 for i in range(self.pop)]
        self.f = [[0 for j in range(self.n_dim)] for i in range(self.pop)]
        self.a = [[0 for j in range(self.n_dim)] for i in range(self.pop)]


        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = [np.inf for i in range(self.pop)]  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

    def cal_q_M(self):
        best = np.min(self.Y)
        worst = np.max(self.Y)
        self.q = (self.Y - worst) / best - worst
        self.M = self.q / sum(self.q)

    def cal_f(self):
        for i in range(self.pop):
            f = None
            for j in range(self.pop):
                if j != i:
                    dividend = float(self.M[i] * self.M[j])
                    temp = self.X[i] - self.X[j]
                    sum_temp = [k**2 for k in temp]
                    divisor = math.sqrt(sum(sum_temp)) + np.finfo('float').eps
                    if f is None:
                        f = self.G * (dividend / divisor) * (self.X[j] - self.X[i])
                    else:
                        f = f + self.G * (dividend / divisor) * (self.X[j] - self.X[i])

            self.f[i] = np.random.uniform(0, 1) * f

    def update_gbest(self):
        idx_min = self.Y.index(min(self.Y))
        if self.gbest_y > self.Y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.Y[idx_min]

    def run(self):
        for iteration in range(self.max_iter):

            self.Y = [self.func(self.X[i]) for i in range(self.pop)]
            self.cal_q_M()
            self.G = self.G * np.e ** (- self.alpha * (iteration / self.max_iter))
            self.cal_f()
            self.a = [self.f[i]/self.M[i] for i in range(self.pop)]
            self.V = (np.random.uniform(0, 1) * self.V) + self.a
            self.update_gbest()
            self.X = self.X + self.V
            self.gbest_y_hist.append(self.gbest_y)
            # print(iteration, self.gbest_x, self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y


def demo_func(args):

    x, y = args[0], args[1]
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


if __name__ == '__main__':
    n_dim = 2
    lb = [0 for i in range(n_dim)]
    ub = [1 for i in range(n_dim)]
    # demo_func = test_function.fu2
    pop_size = 20
    max_iter = 100
    res = []
    for i in range(100):
        pso = GSA(func=demo_func, n_dim=n_dim, pop=pop_size, max_iter=max_iter, lb=lb, ub=ub)
        best_x, bext_y = pso.run()
        print(f'{i}: {demo_func(pso.gbest_x)}\t{pso.gbest_x}')
        res.append(bext_y)
    print(sum(res)/len(res))
    print(np.std(res))
    # plt.plot(pso.gbest_y_hist)
    #
    # plt.show()
