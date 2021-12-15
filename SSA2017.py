# ！usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/12/5 15:39
# @Author : LucXiong
# @Project : Model
# @File : SSA2017.py

############################################################################

# Ref: Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems.pdf

############################################################################

# Required Libraries
import numpy  as np
import math
import random
import os
import matplotlib.pyplot as plt
import test_function

class salp_swarm_algorithm():
    def __init__(self, pop_size=50, n_dim=2, max_iter=150, lb=[-5,-5], ub=[5,5], func=None):
        self.pop = pop_size
        self.lb = lb
        self.ub = ub
        self.func = func
        self.n_dim = n_dim
        self.max_iter = max_iter


        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = [np.inf for i in range(self.pop)]  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_pbest()
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



    # Function: Updtade Position
    def update_position(self, c1):
        for i in range(0, self.pop):
            if (i <= self.pop / 2): # 领导者比例
                for j in range(0, self.n_dim):
                    c2 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
                    c3 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
                    if (c3 >= 0.5):  # c3 < 0.5
                        try:
                            self.X[i, j] = np.clip((self.gbest_x[0][j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])), self.lb[j], self.ub[j])
                        except:
                            self.X[i, j] = np.clip((self.gbest_x[j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])), self.lb[j], self.ub[j])
                    else:
                        try:
                            self.X[i, j] = np.clip((self.gbest_x[0][j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])), self.lb[j], self.ub[j])
                        except:
                            self.X[i, j] = np.clip((self.gbest_x[j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])), self.lb[j], self.ub[j])
            else: # 追随者比例
                for j in range(0, self.n_dim):
                    self.X[i, j] = np.clip(((self.X[i - 1, j] + self.X[i, j]) / 2), self.lb[j], self.ub[j])
        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles

    def run(self):
        for i in range(self.max_iter):
            c1 = 2 * math.exp(-(4 * ((i+1) / self.max_iter)) ** 2)
            self.update_position(c1)
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y


if __name__ == '__main__':
    n_dim = 30
    lb = [-5 for i in range(n_dim)]
    ub = [5 for i in range(n_dim)]
    demo_func = test_function.fu1
    ssa = salp_swarm_algorithm(pop_size=50, n_dim=n_dim, max_iter=150, lb=lb, ub=ub, func=demo_func)
    ssa.run()
    print('best_x is ', ssa.gbest_x, 'best_y is', ssa.gbest_y)
    print(f'{demo_func(ssa.gbest_x)}\t{ssa.gbest_x}')
    plt.plot(ssa.gbest_y_hist)
    plt.show()



