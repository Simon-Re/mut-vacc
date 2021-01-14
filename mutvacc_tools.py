import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sympy import solve, symbols
from sympy.functions import exp
from scipy.special import lambertw
import time

def distance(l1,l2):
    d = 0
    for i in range(len(l1)):
        if l1[i] is not l2[i]:
            d += 1
    return d

class mutvaccmodel:
    def __init__(self):
        self.N = 10000
        self.beta = 0.18
        self.gamma = 1./14
        self.theta = 0.002
        self.theta2 = self.theta
        self.delta = 0.01
        self.p = 0.00001
        self.m = self.p
        self.mu = 1./180
        self.k = 0.2
        self.vacctime = 100
        self.hesitants = 0.1
        self.small_fraction = 10./self.N
        self.measures = True
        self.lockdown_k = [0.01,0.001]
        self.beta2 = self.beta/3
        self.beta3 = self.beta*2/3.
        self.ext_value = 0.01
        
    def setup(self,n):
        genes = '01'
        strains = []
        for g in product(genes,repeat=n):
            strains.append("".join(g))
        
        A = np.zeros((len(strains),len(strains))) #mutation network
        B = np.zeros((len(strains),n)) #resistance matrix
        C = np.zeros(((n+1,1))) #vaccination matrix
        mutation_dict = {}
        
        for obj1 in list(enumerate(strains)):
            mutation_dict[obj1[0]] = []
            for obj2 in list(enumerate(strains)):
                if distance(obj1[1],obj2[1]) == 1:
                    A[obj1[0],obj2[0]] = 1
                    mutation_dict[obj1[0]].append(obj2[0])
            B[obj1[0]] = list(obj1[1])
            
        C[0][0] = -n - 1
        C = C +1
        self.mut_from_to = mutation_dict
        self.A = A
        self.C= np.append(C,np.zeros((n+1,n)),1)
        self.M = A - n*np.identity(len(strains))
        self.B = np.append(np.array(np.ones((len(strains),1))),B,1)
        self.strains = strains
        self.n = n
        self.prepare_fixation_p()
        
    def prepare_fixation_p(self):
        fitme = []
        x = np.linspace(1.01,5.,50)
        for rt in x:
            res = (lambertw(-np.exp(-rt)*rt) + rt)/rt
            #print rt, res
            fitme.append(res.real)

        poly = np.polyfit(x, np.array(fitme,dtype='float64'), 10)
        self.fixation_polynomial = np.poly1d(poly)
        
    def fixation_p(self,Rt):
        res = np.copy(Rt)
        for i in range(len(res)):
            if Rt[i] < 1.:
                res[i] = 0.
            elif Rt[i] > 5.:
                res[i] = 1.
            else:
                res[i] = self.fixation_polynomial(Rt[i])
        return res
    
    def fixation2(self, Rt, strain):
        new = np.random.poisson(self.Nt[strain]*Rt)
        self.Nt[strain] += new -1
        if self.Nt[strain] > self.small_fraction*self.N:
            res = 1.
        else:
            res = 0.
        return res
        
    def beta_ld(self,infected):
        
        if not self.measures:
            return self.beta
        
        if infected > self.lockdown_k[0] or self.lockdown:
            self.lockdown = 1
            self.openup = 0
            b = self.beta2
        if not self.openup and not self.lockdown:
            b = self.beta
        if (self.lockdown and infected < self.lockdown_k[1]) or self.openup:
            self.openup = 1
            self.lockdown = 0
            b = self.beta3
        return b
    
    def vaccspeed(self,S0,R0,I0):
        x = S0 + R0 + I0
        if x <= self.hesitants:
            return 0.
        y = S0 + R0 + 0.1
        return (1.-self.hesitants/x)/y
        
    def run_deterministic(self,T, deltat):
        switch = 0
        D= []
        R, S = [], []
        I = []
        Time = []
        emergetimes = []
        cases = []
        rep_num = []
        I.append(np.zeros((len(self.strains), self.n+1)))
        S.append(np.zeros((self.n+1)))
        R.append(np.zeros((self.n+1)))
        I[0][0,0] = self.small_fraction
        S[0][0] = (1-I[0][0,0])
        D.append(0)
        cases.append(I[0][0,0])
        rep_num.append(self.beta/(self.gamma+self.delta)*(np.matmul(self.B,S[0])))
        Time.append(0)

        for t in range(T-1):
            Inew = I[-1] + S[-1]*(I[-1].sum(1)*self.B.T).T*self.beta*deltat \
                    - I[-1]*self.gamma*deltat - I[-1]*self.delta*deltat \
                    + self.m*np.matmul(self.M,I[-1])*deltat 
            Dnew = D[-1] + sum(sum(I[-1]))*self.delta*deltat
            if cases[-1] > self.k or switch:
                switch = 1
                Rnew = R[-1] - self.mu*R[-1]*deltat + sum(I[-1])*self.gamma*deltat\
                        + self.theta2/self.n*np.matmul(self.C,R[-1])*deltat 
                Snew = S[-1] + self.mu*R[-1]*deltat \
                    + self.theta/self.n*np.matmul(self.C,S[-1])*deltat \
                    - S[-1]*np.matmul(np.transpose(self.B),np.sum(I[-1],1))*self.beta*deltat
            else:
                Rnew = R[-1] - self.mu*R[-1]*deltat + sum(I[-1])*self.gamma*deltat
                Snew = S[-1] + self.mu*R[-1]*deltat \
                    - S[-1]*np.matmul(np.transpose(self.B),np.sum(I[-1],1))*self.beta*deltat
            S.append(Snew)
            I.append(Inew)
            R.append(Rnew)
            D.append(Dnew)
            rep_num.append(self.beta/(self.gamma+self.delta)*(np.matmul(self.B,S[-1])))
            cases.append(sum(sum(I[-1])) + sum(R[-1]) + D[-1])
            Time.append(Time[-1]+deltat)
            
        print 'total' , sum(sum(Inew)) + sum(Rnew + Snew) + Dnew
    
        self.total_dead = D[-1]

        self.graphs = {'S':S,'I':I,'R':R,'D':D,'cases':cases,
                'Time':Time,'Rt':rep_num,'te':emergetimes}
        
    def run_stochastic(self,T, deltat):
        self.started_vacc = 0
        self.lockdown = 0
        self.openup = 0
        
        total = [1.]
        D= []
        R, S = [], []
        I, Nt,Itot = [], [], []
        Time = []
        emergetimes = []
        losstimes = []
        for l in range(len(self.strains)):
            emergetimes.append([])
            losstimes.append([])
        emergetimes[0].append(0.)
        
        cases = []
        rep_num = []
        risk = []
        
        I.append(np.zeros((len(self.strains), self.n+1)))
        Itot.append(np.zeros((len(self.strains), self.n+1)))
        Nt.append(np.zeros((len(self.strains), self.n+1)))
        S.append(np.zeros((self.n+1)))
        R.append(np.zeros((self.n+1)))
        D.append(np.zeros((len(self.strains))))
        I[0][0,0] = self.small_fraction
        S[0][0] = (1-I[0][0,0])
        cases.append(I[0][0,0])
        beta = self.beta_ld(np.sum(I[0]))
        rep_num.append(beta/(self.gamma+self.delta)*(
                np.matmul(self.B,S[0])))
        risk.append(1.-np.exp(-self.p*self.fixation_p(rep_num[0])*np.matmul(self.A,I[0].sum(1))))
        Time.append(0)
        
        thistime = 0
        thattime = 0
        
        for t in range(T-1):
            start = time.time()
            beta = self.beta_ld(np.sum(Itot[-1]))
            
            Inew = I[-1] + S[-1]*(I[-1].sum(1)*self.B.T).T*beta*deltat \
                    - I[-1]*self.gamma*deltat - I[-1]*self.delta*deltat 
                    
            Dnew = D[-1] + I[-1].sum(1)*self.delta*deltat
            
            if cases[-1] > self.k or self.started_vacc or (Time[-1] > self.vacctime and self.openup):
                
                self.started_vacc = 1
                v_vacc = self.vaccspeed(S[-1][0],R[-1][0],sum(Itot[-1][:,0]))
                
                Rnew = R[-1] - self.mu*R[-1]*deltat + sum(I[-1])*self.gamma*deltat \
                        + self.theta2/self.n*self.C.T[0]*deltat*v_vacc*R[-1][0]
                        #+ theta2/n*np.matmul(C,R[-1])*deltat 
                        
                Snew = S[-1] + self.mu*R[-1]*deltat \
                        - S[-1]*np.matmul(np.transpose(self.B),
                           np.sum(I[-1],1))*beta*deltat \
                        + self.theta/self.n*self.C.T[0]*deltat*v_vacc*S[-1][0]
                        #+ theta/n*np.matmul(C,S[-1])*deltat \
            else:
                Rnew = R[-1] - self.mu*R[-1]*deltat + sum(I[-1])*self.gamma*deltat
                Snew = S[-1] + self.mu*R[-1]*deltat \
                    - S[-1]*np.matmul(np.transpose(self.B),
                       np.sum(I[-1],1))*beta*deltat
            
            end = time.time()
            
            thistime += end - start
            
            start = time.time()
            
            Ntnew = np.copy(Nt[-1])
            Inew[Inew < 0] = 0.
            #mutation
            for orig_strain in range(len(self.strains)):
                for vacc in range(self.n+1):
                    n_mut = np.random.poisson(self.N*I[-1][orig_strain,vacc]*self.p)
                    choice_loci = np.random.randint(self.n,size = n_mut)
                    if n_mut > self.N*I[-1][orig_strain,vacc]:
                        n_mut = int(I[-1][orig_strain,vacc])
                    for mutation in range(n_mut):
                        choice_strain = self.mut_from_to[orig_strain][choice_loci[mutation]]
                        if sum(I[-1][choice_strain,:]) == 0.:
                            Ntnew[choice_strain][vacc] += 1.
                        else:
                            Inew[choice_strain][vacc] += 1./self.N
                        Inew[orig_strain][vacc] -= 1./self.N
                        emergetimes[choice_strain].append(t*deltat)
            
            Inew[Inew <0] = 0.
            #micro poisson evolution
            for strain in range(len(self.strains)):
                
                allevents = deltat*(self.gamma +
                                    self.delta + 
                                    beta*np.sum(S[-1]*self.B[strain]))
                options = beta*S[-1]*self.B[strain]*deltat
                options = np.append(options,self.gamma*deltat)
                options = np.append(options,self.delta*deltat)
                options = options/sum(options)
                    
                #Inceftion Recovery
                for vaccine in range(self.n + 1):
                    if Nt[-1][strain,vaccine] >0:
                        n_event = np.random.poisson(Nt[-1][strain,vaccine]*allevents)
                        for event in range(n_event):
                            choice = np.random.choice(range(self.n +3), p=options)
                            if choice <= self.n:
                                Ntnew[strain,choice] += 1
                                Snew[choice] -= 1./self.N
                            if choice == self.n + 1:
                                Ntnew[strain,vaccine] -= 1
                                Rnew[vaccine] += 1./self.N
                            if choice == self.n + 2:
                                Ntnew[strain,vaccine] -= 1
                                Dnew[strain] += 1./self.N
                    Ntnew[Ntnew <= 0] = 0.
                    Snew[Snew <= 0] = 0.
                                
                #FIXATION
                extinction_prob1 = np.exp(-sum(Nt[-1][strain,:])*(self.gamma))
                extinction_prob2 = np.exp(-sum(Itot[-1][strain,:])*self.N*(self.gamma))
                
                if 0. < extinction_prob1 < 0.01:
                    Inew[strain,:] += Nt[-1][strain,:]/self.N
                    Ntnew[strain,:] = 0.
                
                #EXTINCTION
                if 1. > extinction_prob2 > 0.1 and extinction_prob1 ==1:
                    Ntnew[strain,:] += (I[-1][strain,:]*self.N).astype(int)
                    Rnew[:] += (I[-1][strain,:]*self.N \
                                - (I[-1][strain,:]*self.N).astype(int))/self.N
                    Inew[strain,:] = 0.
                  
            end = time.time()
                
            thattime += end - start
            rep_num.append(beta/(self.gamma+self.delta)*(
                    np.matmul(self.B,Snew)))
            Nt.append(Ntnew)
            S.append(Snew)
            I.append(Inew)
            Itot.append(Inew + Ntnew/self.N)
            R.append(Rnew)
            D.append(Dnew)
            risk.append(1.-np.exp(-self.p*self.fixation_p(rep_num[-1])*np.matmul(self.A,Itot[-1].sum(1))))
            cases.append(sum(sum(Itot[-1])) + sum(R[-1]) + sum(D[-1]))
            Time.append(Time[-1]+deltat)
            total.append(sum(sum(Inew + Ntnew/self.N))
                                + sum(Rnew + Snew) + sum(Dnew))
            if (Inew+Nt>0).sum() == 0:
                break

       # print np.array([thistime, thattime])/(thistime + thattime)
        #print 'total' , sum(sum(Inew + Ntnew/self.N)) + sum(Rnew + Snew) + sum(Dnew)
        self.total_dead = sum(D[-1])
        
        self.graphs = {'S':S,'Itot':Itot,'I':I,'R':R,'D':D,'cases':cases,
                'Time':Time,'Rt':rep_num,'te':emergetimes,'tl':losstimes,
                'Nt': Nt, 'risk':risk,'total':total}
        return self.graphs
    
    #%%
    def plot_me(self,plot_lines):
        
        
        plt.plot(self.graphs['Time'],self.graphs['total'])
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('total',size = 20)
        plt.legend(range(self.n+1),loc='best')
        plt.show()
        
        plt.plot(self.graphs['Time'],self.graphs['S'])
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('fractions S and V',size = 20)
        plt.legend(range(self.n+1),loc='best')
        plt.show()

        plt.plot(self.graphs['Time'],self.graphs['R'])
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('fractions R',size = 20)
        plt.legend(range(self.n+1),loc='best')
        plt.show()
        
        plt.plot(self.graphs['Time'],self.graphs['cases'])
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('cases',size = 20)
        plt.legend(['cases'],loc='best')
        plt.show()
        
        plt.plot(self.graphs['Time'],np.array(self.graphs['Itot']).sum(2))
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('fractions Itot types',size = 20)
        plt.legend(self.strains,loc='best')
        if plot_lines:
            for l in self.graphs['te']:
                for e in l:
                    plt.axvline(x=e,color='r',alpha=0.5)
            for l in self.graphs['tl']:
                for e in l:
                    plt.axvline(x=e,color='k',alpha=0.5)
        plt.show()

        plt.plot(self.graphs['Time'],np.array(self.graphs['I']).sum(2))
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('fractions I types',size = 20)
        plt.legend(self.strains,loc='best')
        if plot_lines:
            for l in self.graphs['te']:
                for e in l:
                    plt.axvline(x=e,color='r',alpha=0.5)
            for l in self.graphs['tl']:
                for e in l:
                    plt.axvline(x=e,color='k',alpha=0.5)
        plt.show()
        
        plt.plot(self.graphs['Time'],np.array(self.graphs['Nt']).sum(2))
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('fractions Nt types',size = 20)
        plt.legend(self.strains,loc='best')
        if plot_lines:
            for l in self.graphs['te']:
                for e in l:
                    plt.axvline(x=e,color='r',alpha=0.5)
            for l in self.graphs['tl']:
                for e in l:
                    plt.axvline(x=e,color='k',alpha=0.5)
        plt.show()
    
        plt.plot(self.graphs['Time'],self.graphs['Rt'])
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('rep nums',size = 20)
        plt.legend(self.strains,loc='best')
        plt.show()

        plt.plot(self.graphs['Time'],self.graphs['D'])
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('fraction Dead',size = 20)
        plt.legend(self.strains,loc='best')
        plt.show()
        
        plt.plot(self.graphs['Time'],self.graphs['risk'])
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('risk',size = 20)
        plt.legend(self.strains,loc='best')
        plt.show()