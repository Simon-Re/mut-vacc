import numpy as np
import matplotlib.pyplot as plt
import mutvacc_tools as mv
import os
import time
import sys
    
def get_task_params(task_id,n1,n2,n3):
    #n1*n2*n3 = len(task_id)
    i1 = np.remainder(task_id,n1)
    i2 = np.remainder(int(task_id/n1),n2)
    i3 = np.remainder(int(task_id/(n1*n2)),n3)
    return i1,i2,i3

class sim:
    def __init__(self):
        self.experiment_title = 'test'
        
        if not os.path.exists(self.experiment_title):
            os.makedirs(self.experiment_title)
           
        self.deltat = 1.
        self.T = int(365/self.deltat)
        n = 1
        self.model = mv.mutvaccmodel()
        self.model.mu = 1./180
        self.model.setup(n)
        self.model.hesitants = 0.1
        self.model.lockdown_k = [0.05,0.001]
        self.model.beta = 0.22
        self.model.beta2 = self.model.beta/3.
        self.model.beta3 = self.model.beta*2.5/3.
        self.model.theta = 0.01
        self.model.theta2 = 0.01
        self.model.vacctime = 100
        self.model.ext_value = 0.1
        self.model.k = 1.
        self.model.p = 0.0001
        self.model.N = 100000
        self.model.small_fraction = 10./self.model.N
        
        self.ax1 = 10
        ft1 = [0.00175,0.00225]
        self.param_space1 = np.linspace(ft1[0],ft1[1],self.ax1)
        self.tit1 = r'$I_{measures}$'
        
        self.ax2 = 10
        ft2 = [0,300]
        self.param_space2 = np.linspace(ft2[0],ft2[1],self.ax2)
        self.tit2 = r'$t_{vacc}$'
        
        self.tit3 = r'p'
        self.ax3 = 3
        self.param_space3 = np.array([0.0001,0.00005,0.00001])
        
        self.iterations = 50
        
        f = open(self.experiment_title+"/parameters.txt", "w")
        for entry in self.model.__dict__:
            if entry in ['theta2','theta', 'hesitants', 
                         'k','m','N','mu','p','beta','delta',
                         'small_fraction','n','gamma']:
                f.write(entry +' ' + str(self.model.__dict__[entry]) + '\n')
        f.write('T ' + str(self.T) + '\n')
        f.write('deltat ' + str(self.deltat))
        f.close()

    def run(self,task_id):
        
        ids = get_task_params(task_id, self.ax1, self.ax2, self.ax3)
        self.model.lockdown_k[0] = self.param_space1[ids[0]]
        self.model.vacctime = self.param_space2[ids[1]]
        self.model.p = self.param_space3[ids[2]]
                
        self.model.run_stochastic(self.T,self.deltat)
                
    def save_results(self):
        emerged = np.heaviside(self.model.graphs['D'][-1],0)
        deaths = self.model.graphs['D'][-1]
        a, b, c = str(self.model.lockdown_k[0]), \
                   str(self.model.vacctime), \
                   str(self.model.p)
        
        with open(self.experiment_title+"/deaths.txt", "a+") as f:
            f.write(a + "\t" +  b + "\t" + c + "\t" +"{}\t{}".format(*deaths) + "\n")
        with open(self.experiment_title+"/emerged.txt", "a+") as f:
            f.write(a + "\t" +  b + "\t" + c + "\t" +"{}\t{}".format(*emerged) + "\n")
        
my_task = int(sys.argv[1])
sim = sim()
sim.run(my_task)
sim.save_results()
