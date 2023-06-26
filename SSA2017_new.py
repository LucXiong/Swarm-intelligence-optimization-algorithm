import numpy as np
import matplotlib.pyplot as plt
import test_function

class SA:
    def __init__(self,iterations,n_dim,lb,ub,pop_size,target_function):
        self.iterations=iterations
        self.n_dim=n_dim
        self.lb=lb
        self.ub=ub
        self.pop_size=pop_size
        self.target_function=target_function

        self.X=np.random.uniform()
        self.X=np.random.uniform(lb,ub,(pop_size,n_dim+1))
        for i in range(len(self.X)):
            self.X[i][-1] = self.target_function(self.X[i][:-1])
        self.X=sorted(self.X,key=lambda x: x[-1])
        self.g_best=self.X[0]
        self.gbest_Y_history=[self.X[0][-1]]

    def run(self,visible=False):
        for iter in range(self.iterations):
            c1=2*np.power(np.e,-(4*iter/self.iterations)**2)
            self.X[0]=self.g_best+np.random.uniform(-1,1)*c1*(np.random.uniform()*(self.ub-self.lb)+self.lb)
            self.X[0][:-1] = np.clip(self.X[0][:-1], self.lb, self.ub)
            self.X[0][-1] = self.target_function(self.X[0][:-1])
            for i in range(1,self.pop_size):
                self.X[i]=0.5*(self.X[i]+self.X[i-1])
                self.X[i][-1] = self.target_function(self.X[i][:-1])
            self.X=sorted(self.X,key=lambda x: x[-1])
            if self.g_best[-1]>self.X[0][-1]:
                self.g_best=self.X[0]
            self.gbest_Y_history.append(self.g_best[-1])
            if visible:
                print(self.X[0])

    def result(self):
        return self.X[0]

    def convergence_curve(self):
        plt.plot(self.gbest_Y_history)
        plt.yscale('log')
        plt.show()

if __name__ == '__main__':
    test=SA(iterations=1000,n_dim=2,lb=-10,ub=10,pop_size=50,target_function=test_function.F6)
    test.run(visible=True)
    print("result:{}".format(test.result()))
    test.convergence_curve()
