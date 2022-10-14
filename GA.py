import copy
import test_function
import matplotlib.pyplot as plt
import random

class GA():

    def __init__(self,iterations,n_dim,lb,ub,pop_size,target_function,retain_rate,random_select_rate,mutation_probability):
        self.iterations=iterations
        self.n_dim=n_dim
        self.lb=lb
        self.ub=ub
        self.pop_size=pop_size
        self.target_function=target_function
        self.retain_rate=retain_rate
        self.random_select_rate=random_select_rate
        self.mutation_probability=mutation_probability

    def particle_init(self):
        self.particle=[[] for i in range(self.pop_size)]
        self.g_best=[0 for i in range(self.n_dim)]
        self.g_best.append(float('inf'))
        for i in range(self.pop_size):
            for j in range(self.n_dim):
                self.particle[i].append(random.uniform(self.lb,self.ub))
            self.particle[i].append(self.target_function(self.particle[i]))
            if self.g_best[-1]>self.particle[i][-1]:
                self.g_best = copy.deepcopy(self.particle[i])
        self.chosen_probability=[0 for i in range(self.pop_size)]
        self.calculate_chosen_probability()

    def calculate_chosen_probability(self):
        fitness=0

        for i in range(self.pop_size):
            fitness+=self.particle[i][-1]
        for i in range(self.pop_size):
            self.chosen_probability[i]=self.particle[i][-1]/fitness

    def selction(self):
        fitness=[]
        for i in self.particle:
            fitness.append(i[-1])
        fitness.sort(reverse=True)
        retain_criteria=fitness[int(self.pop_size*self.retain_rate)]
        parents=[]
        for i in range(self.pop_size):
            if self.particle[i][-1]<=retain_criteria or random.random()<self.random_select_rate:
                parents.append(self.particle[i])
        self.particle=copy.deepcopy(parents)

    def cross(self):
        count_parents=len(self.particle)
        count_cross=self.pop_size-count_parents
        for i in range(count_cross):
            parent_1 = self.particle[random.randint(0, count_parents-1)]
            parent_2 = self.particle[random.randint(0, count_parents-1)]
            child=[]
            for j in range(self.n_dim):
                child.append(random.uniform(min(parent_1[j], parent_2[j]), max(parent_1[j], parent_2[j])))
            child.append(self.target_function(child))
            self.particle.append(child)
            if self.g_best[-1]>self.particle[-1][-1]:
                self.g_best=copy.deepcopy(self.particle[-1])


    def mutate(self,iteration):
        for i in range(int(self.pop_size*self.retain_rate),self.pop_size):
            if random.random()<self.mutation_probability:
                for j in range(self.n_dim):
                    self.particle[i][j]+=random.uniform(-0.1,0.1)*(self.ub-self.lb)*(1-iteration/self.iterations)
                    if self.particle[i][j]>self.ub:
                        self.particle[i][j]=self.ub
                    if self.particle[i][j]<self.lb:
                        self.particle[i][j]=self.lb
                self.particle[i][-1]=self.target_function(self.particle[i][:self.n_dim])
                if self.g_best[-1]>self.particle[i][-1]:
                    self.g_best=copy.deepcopy(self.particle[i])

    def run(self):
        self.particle_init()
        self.g_best_hist=[self.g_best[-1]]
        for i in range(self.iterations):
            self.calculate_chosen_probability()
            self.selction()
            self.cross()
            self.mutate(i)
            self.g_best_hist.append(self.g_best[-1])

    def result(self):
        return self.g_best

    def convergence_curve(self):
        plt.plot(self.g_best_hist)
        plt.yscale('log')
        plt.show()

if __name__ == '__main__':

    test=GA(iterations=1000,n_dim=1,lb=-500,ub=500,pop_size=50,target_function=function,retain_rate=0.3,random_select_rate=0.2,mutation_probability=0.8)
    test.run()
    result=test.result()
    test.convergence_curve()
    print(result)