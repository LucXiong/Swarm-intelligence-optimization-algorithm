# ！usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/12/2 11:10
# @Author : LucXiong
# @Project : Model
# @File : SSA.py

"""
Ref:https://github.com/changliang5811/SSA_python
Ref:https://www.tandfonline.com/doi/full/10.1080/21642583.2019.1708830
Ref:A novel swarm intelligence optimization approach: sparrow search algorithm.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import random
import test_function

class SSA():
    def __init__(self, func, n_dim=None, pop_size=20, max_iter=50, lb=-512, ub=512, verbose=False):
        self.func = func
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func
        self.pop = pop_size  # number of particles
        P_percent = 0.2 # # 生产者的人口规模占总人口规模的20%
        D_percent = 0.1 # 预警者的人口规模占总人口规模的10%
        self.pNum = round(self.pop * P_percent)  # 生产者的人口规模占总人口规模的20%
        self.warn = round(self.pop * D_percent)  # 预警者的人口规模占总人口规模的10%

        self.max_iter = max_iter  # max iter
        self.verbose = verbose  # print the result of each iter or not

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
        self.update_pbest()
        self.update_gbest()
        #
        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated
        self.idx_max = 0
        self.x_max = self.X[self.idx_max, :]
        self.y_max = self.Y[self.idx_max]

    def cal_y(self, start, end):
        for i in range(start, end):
            self.Y[i] = self.func(self.X[i])

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

    def find_worst(self):
        self.idx_max = self.Y.index(max(self.Y))
        self.x_max = self.X[self.idx_max, :]
        self.y_max = self.Y[self.idx_max]

    def update_finder(self, iter_num):
        r2 = np.random.rand(1)  # 预警值
        self.idx = sorted(enumerate(self.Y), key=lambda x: x[1])
        self.idx = [self.idx[i][0] for i in range(len(self.idx))]
        # 这一部位为发现者（探索者）的位置更新
        if r2 < 0.8:  # 预警值较小，说明没有捕食者出现
            for i in range(self.pNum):
                r1 = np.random.rand(1)
                self.X[self.idx[i], :] = self.X[self.idx[i], :] * np.exp(-(iter_num) / (r1 * self.max_iter))  # 对自变量做一个随机变换
                self.X = np.clip(self.X, self.lb, self.ub) # 对超过边界的变量进行去除
                # X[idx[i], :] = Bounds(X[idx[i], :], lb, ub)  # 对超过边界的变量进行去除
                # fit[sortIndex[0, i], 0] = func(X[sortIndex[0, i], :])  # 算新的适应度值
        elif r2 >= 0.8:  # 预警值较大，说明有捕食者出现威胁到了种群的安全，需要去其它地方觅食
            for i in range(self.pNum):
                # Q = np.random.rand(1)  # 也可以替换成
                Q = np.random.normal(loc=0, scale=1.0, size=1)
                self.X[self.idx[i], :] = self.X[self.idx[i], :] + Q * np.ones((1, self.n_dim))  # Q是服从正态分布的随机数。L表示一个1×d的矩阵
                self.X = np.clip(self.X, self.lb, self.ub)  # 对超过边界的变量进行去除
                # X[idx[i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
                # fit[sortIndex[0, i], 0] = func(X[sortIndex[0, i], :])
        self.cal_y(0, self.pNum)

    def update_follower(self):
        #  这一部位为加入者（追随者）的位置更新
        for ii in range(self.pop - self.pNum):
            i = ii + self.pNum
            A = np.floor(np.random.rand(1, self.n_dim) * 2) * 2 - 1
            best_idx = self.Y[0:self.pNum].index(min(self.Y[0:self.pNum]))
            bestXX = self.X[best_idx, :]
            if i > self.pop/2:
                Q = np.random.rand(1)
                self.X[self.idx[i],:] = Q*np.exp((self.x_max-self.X[self.idx[i],:])/np.square(i))
            else:
                self.X[self.idx[i],:] = bestXX+np.dot(np.abs(self.X[self.idx[i],:]-bestXX),1/(A.T*np.dot(A,A.T)))*np.ones((1, self.n_dim))
        self.X = np.clip(self.X, self.lb, self.ub)  # 对超过边界的变量进行去除
        self.cal_y(self.pNum, self.pop)

    def detect(self):
        arrc = np.arange(self.pop)
        c = np.random.permutation(arrc)  # 随机排列序列
        b = [self.idx[i] for i in c[0: self.warn]]
        e = 10e-10
        for j in range(len(b)):
            if self.Y[b[j]] > self.gbest_y:
                self.X[b[j], :] = self.gbest_x + norm.rvs(size=self.n_dim) * np.abs(self.X[b[j], :] - self.gbest_x)
            else:
                self.X[b[j], :] = self.X[b[j], :]+(2*np.random.rand(1)-1)*np.abs(self.X[b[j], :]-self.x_max)/(self.func(self.X[b[j]])-self.y_max+e)
            self.X = np.clip(self.X, self.lb, self.ub)
            self.Y[b[j]] = self.func(self.X[b[j]])

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            self.update_finder(iter_num) # 更新发现者位置
            self.find_worst()  # 取出最大的适应度值和最差适应度的X
            self.update_follower() # 更新跟随着位置
            self.update_pbest()
            self.update_gbest()
            self.find_worst()  # 取出最大的适应度值和最差适应度的X
            self.detect()
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        return self.best_x, self.best_y

if __name__ == '__main__':
    n_dim = 30
    lb = [-100 for i in range(n_dim)]
    ub = [100 for i in range(n_dim)]
    demo_func = test_function.fu5
    pop_size = 100
    max_iter = 100
    ssa = SSA(demo_func, n_dim=n_dim, pop_size=pop_size, max_iter=max_iter, lb=lb, ub=ub)
    ssa.run()
    print('best_x is ', ssa.gbest_x, 'best_y is', ssa.gbest_y)
    # print(f'{demo_func(ssa.gbest_x)}\t{ssa.gbest_x}')
    plt.plot(ssa.gbest_y_hist)
    plt.show()
