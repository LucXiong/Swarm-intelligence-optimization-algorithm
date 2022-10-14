import matplotlib.pyplot as plt
import random
import copy
import test_function

class DE():
    def __init__(self, iterations, n_dim, lb, ub, pop_size, target_function,mutate_factor,cross_rate):
        self.iterations = iterations
        self.n_dim = n_dim
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.target_function = target_function
        self.mutate_factor=mutate_factor
        self.cross_rate=cross_rate

    def particle_init(self):
        self.particles = [[] for i in range(self.pop_size)]
        for i in range(self.pop_size):
            for j in range(self.n_dim):
                self.particles[i].append(random.uniform(self.lb, self.ub))
            self.particles[i].append(self.target_function(self.particles[i]))
            if self.g_best[-1] > self.particles[i][-1]:
                self.g_best = copy.deepcopy(self.particles[i])

    def mutate_and_cross(self):
        for i in range(self.pop_size):
            target_1=random.randint(0,self.pop_size-1)
            while(target_1==i):
                target_1=random.randint(0,self.pop_size-1)
            target_2=random.randint(0,self.pop_size-1)
            while(target_2==i or target_2==target_1):
                target_2 = random.randint(0, self.pop_size-1)
            target_3=random.randint(0,self.pop_size-1)
            while(target_3==i or target_3==target_1 or target_3==target_2):
                target_3 = random.randint(0, self.pop_size-1)
            target=[]
            for j in range(self.n_dim):
                target.append(self.particles[target_1][j]+self.mutate_factor*(self.particles[target_2][j]-self.particles[target_3][j]))
                if random.random()>self.cross_rate:
                    target[j]=self.particles[i][j]
                if target[j]<self.lb:
                    target[j]=self.lb
                if target[j]>self.ub:
                    target[j]=self.ub
            target.append(self.target_function(target))
            if self.particles[i][-1]>target[-1]:
                self.particles[i]=target
            if self.g_best[-1] > self.particles[i][-1]:
                self.g_best = copy.deepcopy(self.particles[i])

    def run(self):
        self.g_best = [0 for i in range(self.pop_size)]
        self.g_best.append(float('inf'))
        self.g_best_hist = []
        self.particle_init()
        for i in range(self.iterations):
            self.mutate_and_cross()
            self.g_best_hist.append(self.g_best[-1])

    def result(self):
        return self.g_best

    def convergence_curve(self):
        plt.plot(self.g_best_hist)
        plt.yscale('log')
        plt.show()

if __name__ == '__main__':
    test = DE(iterations=500, n_dim=30, lb=-500, ub=500, pop_size=50, target_function=test_function.F1, mutate_factor=0.3,cross_rate=0.5)
    test.run()
    result = test.result()
    test.convergence_curve()
    print(result)