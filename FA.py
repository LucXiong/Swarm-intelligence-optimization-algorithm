# ！usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/12/15 15:02
# @Author : LucXiong
# @Project : Model
# @File : FA.py

"""
Ref:https://github.com/GoodLittleStar/Fireworks/blob/master/FireWork.py
Ref:Tan Y, Zhu Y. Fireworks Algorithm for Optimization[M].
Lecture Notes in Computer Science. City: Springer Berlin Heidelberg, 2010: 355-64[2021-12-08T08:42:21].
"""

import random  # random Function
import numpy as np # numpy operations
import copy
import matplotlib.pyplot as plt
import math
import test_function

class FA():
    def __init__(self, pop_size=50, n_dim=2, m=50, a=0.04, b=0.8, A=40, lb=-1e5, ub=1e5, max_iter=1000, func=None):

        self.a = a
        self.b = b
        self.m = m
        self.A = A
        self.pop = pop_size
        self.dim = n_dim
        self.func = func
        self.max_iter = max_iter  # max iter
        self.epsino = 1e-6
        self.mutate = 0.1

        self.lb, self.ub = np.array(lb) * np.ones(self.dim), np.array(ub) * np.ones(self.dim)
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        self.X = self.X.tolist()
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles

        # self.pbest_x = self.X.copy()  # personal best location of every particle in history
        # self.pbest_y = [np.inf for i in range(self.pop)]  # best image of every particle in history
        # self.fit = [1 / (1 + self.Y[i]) if self.Y[i] > 0 else 1 - self.Y[i] for i in range(self.pop)]
        # self.prob = [self.fit[i] / sum(self.fit) for i in range(self.pop)]
        self.bestindex = self.Y.index(min(self.Y))
        self.gbest_x = self.X[self.bestindex]
        self.gbest_y = min(self.Y) # global best y for all particles
        self.gbest_y_hist = [self.gbest_y]  # gbest_y of every iteration
        # self.update_gbest()


    # def update_pbest(self):
    #     '''
    #     personal best
    #     :return:
    #     '''
    #     for i in range(len(self.Y)):
    #         if self.pbest_y[i] > self.Y[i]:
    #             self.pbest_x[i] = self.X[i]
    #             self.pbest_y[i] = self.Y[i]
    #
    # def update_gbest(self):
    #     '''
    #     global best
    #     :return:
    #     '''
    #     idx_min = self.pbest_y.index(min(self.pbest_y))
    #     if self.gbest_y > self.pbest_y[idx_min]:
    #         self.gbest_x = self.X[idx_min, :].copy()
    #         self.gbest_y = self.pbest_y[idx_min]

    def CalculateSi(self):
        self.MaxFitness = max(self.Y)
        temp = 0.
        self.Si = []
        for i in range(0, self.pop):
            temp = temp + self.MaxFitness - self.Y[i]
        for i in range(0, self.pop):
            self.Si.append(self.m * (self.MaxFitness - self.Y[i] + self.epsino) / (temp + self.epsino))
            if self.Si[-1] < self.a * self.m:
                self.Si[-1] = round(self.a * self.m)
            elif self.Si[-1] > self.b * self.m:
                self.Si[-1] = round(self.b * self.m)
            else:
                self.Si[-1] = round(self.Si[-1])

    def CalculateExpo(self):
        self.MinFitness = min(self.Y)
        temp = 0.
        self.Ai = []
        for i in range(self.pop):
            temp = temp + self.Y[i] - self.MinFitness
        for i in range(self.pop):
            self.Ai.append(self.A * (self.Y[i]- self.MinFitness + self.epsino) / (temp + self.epsino))

    def Explosion(self):
        for k in range(0, self.pop):
            for i in range(self.Si[k]):
                spark = copy.deepcopy(self.X[k])
                z = round(self.dim * random.uniform(0, 1))
                dim_list = range(self.dim)
                rand_z = random.sample(dim_list, z)
                h = self.Ai[k] * random.uniform(-1, 1)
                for j in rand_z:
                    spark[j] += h
                    if spark[j] < self.lb[j] or spark[j] > self.ub[j]:
                        spark[j] = self.lb[j] + abs(spark[j]) % (self.ub[j] - self.lb[j])
                self.X.append(spark)
                self.Y.append(self.func(spark))
            if(len(self.X) > 5 * self.pop):
                break

    def Mutation(self):
        currentsize = len(self.X)
        for k in range(round(self.mutate * currentsize)):
            randindex = random.randint(0, currentsize - 1)
            spark = copy.deepcopy(self.X[randindex])
            # print(spark)
            # print(randindex)
            z = round(self.dim * random.uniform(0, 1))
            dim_list = range(self.dim)
            rand_z = random.sample(dim_list, z)
            g = random.gauss(1, 1)
            for j in rand_z:
                spark[j] *= g
                if spark[j] < self.lb[j] or spark[j] > self.ub[j]:
                    spark[j] = self.lb[j] + abs(spark[j]) % (self.ub[j] - self.lb[j])
            self.X.append(spark)
            self.Y.append(self.func(spark))
            if (len(self.X) > 10 * self.pop):
                break

    def Selection(self):
        newpop=[]
        newpop.append(self.gbest_x)
        self.Ri = []
        for i in range(len(self.X)):
            dis=0.
            for j in range(len(self.X)):
                for k in range(self.dim):
                    dis+= (self.X[i][k]-self.X[j][k])**2
            self.Ri.append(math.sqrt(dis))
        sr = sum(self.Ri)
        px = [self.Ri[i]/sr for i in range(len(self.Ri))]
        for i in range(self.pop-1):
            rr=random.uniform(0,1)
            index=0
            for j in range(self.pop):
                if j==0 and rr<px[j]:
                    index=j
                elif rr>=px[j] and rr<px[j+1]:
                    index=j+1
            newpop.append(self.X[index])
        self.X = newpop
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]

    def run(self):
        for i in range(self.max_iter):
            # print(i)
            # print(len(self.X))
            self.CalculateSi()
            self.CalculateExpo()
            self.Explosion()
            # print(len(self.X))
            self.Mutation()
            # print(len(self.X))
            bestindex = self.Y.index(min(self.Y))
            if self.gbest_y_hist[-1] > self.Y[bestindex]:
                self.gbest_y_hist.append(self.Y[bestindex])
                self.gbest_x = self.X[bestindex]
            else:
                self.gbest_y_hist.append(self.gbest_y_hist[-1])
            self.Selection()
            # print(self.gbest_y_hist[-1])
        return self.gbest_x, self.gbest_y_hist[-1]

if __name__ == '__main__':
    n_dim = 30
    lb = [-100 for i in range(n_dim)]
    ub = [100 for i in range(n_dim)]
    demo_func = test_function.fm2
    pop_size = 100
    max_iter = 200
    fa = FA(n_dim=n_dim, pop_size=pop_size, max_iter=max_iter, lb=lb, ub=ub, func=demo_func)
    best_x, bext_y = fa.run()
    print(f'{demo_func(fa.gbest_x)}\t{fa.gbest_x}')
    plt.plot(fa.gbest_y_hist)
    plt.show()
