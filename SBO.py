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
                # print(f'k is {k}')
                # print(self.X[i,:])
                # print(self.X[i, k])
                # print(self.gbest_x)
                # print(self.gbest_x[0][k])
                try:
                    self.X[i, k] += lemta * 0.5 * (self.X[j, k] + self.gbest_x[0][k] - 2 * self.X[i, k])
                except:
                    self.X[i, k] += lemta * 0.5 * (self.X[j, k] + self.gbest_x[k] - 2 * self.X[i, k])

            # 变异
            if np.random.rand(1) < self.r_mutate and i != idx_min:
                for j in range(self.n_dim):
                    # 正态分布 Satin bowerbird optimizer: A new optimization algorithm to optimize ANFIS for software development effort estimation.pdf
                    self.X[i][j] += self.z * np.random.normal() * (self.ub[j] - self.lb[j])
        self.X = np.clip(self.X, self.lb, self.ub)
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]


    def mutate(self):
        idx_min = self.Y.index(min(self.Y))
        for i in range(self.pop):
            # todo(xionglei@sjtu.edu.cn): 补充判断条件，不是当前最优值可变异
            if np.random.rand(1)[0] < self.r_mutate and i != idx_min:
                for j in range(self.n_dim):
                    self.X[i][j] += self.z * np.random.normal() * (self.ub[j] - self.lb[j])
        self.X = np.clip(self.X, self.lb, self.ub)
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]


    def run(self):
        for i in range(self.max_iter):
            print(i)
            self.update(i)
            # self.mutate() # 不使用最好的1个去变异
            self.cal_prob()
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

def demo_func(x): # Eggholder Function
    x1 = x[0]
    x2 = x[1]
    result = -(x2 + 47) * math.sin(math.sqrt(math.fabs(x2 + x1/2 + 47))) - x1 * math.sin(math.sqrt(math.fabs(x1 - x2 - 47)))
    return result

def demo_func2(x): # Eggholder Function
    result = 0
    for i in range(len(x)):
        result += (x[i] + 0.5) ** 2
    return result

def demo_func3(x): # Eggholder Function
    result = 0
    for i in range(len(x)):
        result += x[i] * x[i]
    return result

def demo_func4(x): # Eggholder Function
    result = 1
    for i in range(len(x)):
        x[i] = abs(x[i])
        result *= x[i]
    result += sum(x)
    return result

def demo_func5(x): # Eggholder Function
    result = 0
    for i in range(len(x)):
        result += (x[i] ** 2 - 10 * math.cos(2 * math.pi * x[i]) + 10)
    return result

def test(demo):
    n_dim = 30
    # demo = demo_func2
    lb = [-100 for i in range(n_dim)]
    ub = [100 for i in range(n_dim)]
    sbo = SBO(n_dim=n_dim, pop_size=50, max_iter=500, lb=lb, ub=ub, func=demo)
    best_x, best_y = sbo.run()
    print(f'{best_x} {best_y}')
    return best_y

if __name__ == '__main__':
    # # lb = [-512, -512]
    # # ub = [512, 512]
    # n_dim =30
    # demo = demo_func3
    # lb = [-5.12 for i in range(n_dim)]
    # ub = [5.12 for i in range(n_dim)]
    # sbo = SBO(n_dim=n_dim, pop_size=50, max_iter=1000, lb=lb, ub=ub, func=demo)
    # # print(sbo.lb)
    # # print(sbo.ub)
    # # print(sbo.X)
    # best_x, bext_y = sbo.run()
    # # print(best_x)
    # # print(bext_y)
    # # print(demo(best_x))
    # # plt.plot(sbo.gbest_y_hist)
    # # plt.show()
    num = 30
    result_list = []
    for demo in [demo_func2, demo_func5]:
        result = []
        for i in range(num):
            result.append(test(demo))
            print(f'{i} {result[-1]}')
        print(result)
        print(sum(result)/num)
        result_list.append(result)

"""
demo_func5
[0.001648592979888619, 4.224274998559954, 5.630406216070336, 2.0035874441366293, 4.036113428893531, 4.276099958806002, 6.001787786574688, 1.1443167870297355, 4.008073473242671, 1.4455340751912633, 3.0138176095887523, 4.081121524114208, 1.0371617678212228, 0.9983023779352695, 3.023934439758598, 5.046781655333252, 4.024030914637036, 2.0158655037790165, 3.024007655952939, 4.026974381082438, 4.117685991683345, 1.0505763605148744, 3.2622530490591526, 1.0756309493968512, 1.2028055341097534, 4.034605981510101, 3.009350829165964, 2.9932117173265382, 2.0076922108639508, 2.035929515920806]
2.928452757701292
demo_func2
[0.00016899879758983439, 0.000213111074694157, 0.00012991674145853316, 0.00017856508309067258, 0.00015425645045409264, 0.00020394856012078633, 0.00024720543859307285, 0.0001223855177818688, 0.00010764582374372907, 0.00018134207406712835, 0.00025414054475744724, 0.000248305314881893, 0.00019060892643053877, 9.98858844725303e-05, 0.00020890254027147896, 0.00016561005970116432, 0.0003638325331999804, 0.00010979405269119638, 0.00028905269223583006, 0.00019417512861607503, 7.633649442728452e-05, 0.00024360837306609159, 0.00011538564858337442, 0.00018062104007293006, 0.00014163446076822093, 0.0002733129206808298, 0.00011916057602143842, 0.00018628894000135584, 0.0001417260572537117, 0.00015444082426112774]
0.0001821399524662791
"""
