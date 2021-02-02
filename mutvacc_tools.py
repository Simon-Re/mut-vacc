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
        self.k = 1.
        self.vacctime = 100
        self.hesitants = 0.01
        self.small_fraction = 10./self.N
        self.measures = True
        self.lockdown_k = [0.01,0.001]
        self.beta2 = self.beta/3
        self.beta3 = self.beta*2/3.
        self.ext_value = 0.01
        self.from_stochastic = 0.01
        self.to_stochastic = 0.1
        self.michaelis_menten = 0.01
        self.smart_lockdown = False
        self.Xc = 10
        self.init_vacc = 0.
        self.init_load = 0.
        
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
        
    def beta_ld(self,infected,t):
        
        th = self.vacctime + 1./self.theta*(1.-self.gamma/self.beta3)
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
        if th - 30 < t < th +30 and self.smart_lockdown:
            self.lockdown = 1
            self.openup = 0
            b = self.beta2
        return b
    
    def vaccspeed(self,S0,R0,I0):
        x = S0 + R0 + I0
        if x <= self.hesitants:
            return 0.
        y = S0 + R0 + self.michaelis_menten
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
        is_extinct = []
        self.e_time = [0.]
        self.tester = 0
        for l in range(len(self.strains)):
            emergetimes.append([])
            losstimes.append([])
            self.e_time.append(T)
            is_extinct.append(1)
        emergetimes[0].append(0.)
        
        cases = []
        rep_num = []
        risk = []
        selection_coeff = []
        
        pfix = [1./self.N]
        print pfix
        I.append(np.zeros((len(self.strains), self.n+1)))
        Itot.append(np.zeros((len(self.strains), self.n+1)))
        Nt.append(np.zeros((len(self.strains), self.n+1)))
        S.append(np.zeros((self.n+1)))
        R.append(np.zeros((self.n+1)))
        gill_time = np.zeros(len(self.strains))
        D.append(np.zeros((len(self.strains))))
        Nt[0][0,0] = int(self.small_fraction*self.N)
        Itot[0][0,0] = self.small_fraction
        S[0][0] = (1-Itot[0][0,0])
        cases.append(Itot[0][0,0])
        beta = self.beta_ld(np.sum(I[0]),0)
        rep_num.append(beta/(self.gamma+self.delta)*(
                np.matmul(self.B,S[0])))
        selection_coeff.append((rep_num[-1]*(self.gamma+self.delta)-beta*S[-1][0])/
                               (beta*S[-1][0]))
        risk.append(1.-np.exp(-self.N*self.p*self.fixation_p(selection_coeff[0]+1.)*np.matmul(self.A,I[0].sum(1))))
        Time.append(0)
        
        thistime = 0
        thattime = 0

        for t in range(T-1):
            start = time.time()
            beta = self.beta_ld(np.sum(Itot[-1]),Time[-1])
            
            Inew = I[-1] + S[-1]*(I[-1].sum(1)*self.B.T).T*beta*deltat \
                    - I[-1]*self.gamma*deltat - I[-1]*self.delta*deltat 
                    
            Dnew = D[-1] + I[-1].sum(1)*self.delta*deltat
            
            if cases[-1] > self.k or self.started_vacc or Time[-1] > self.vacctime:
                
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
            
            #MUTATION
            for orig_strain in range(len(self.strains)):
                
                for vacc in range(self.n+1):
                    
                    n_mut = np.random.poisson(self.N*I[-1][orig_strain,vacc]*self.p*deltat)
                    n_mut2 = np.random.poisson(Ntnew[orig_strain,vacc]*self.p*deltat)
                    choice_loci = np.random.randint(self.n,size = n_mut)
                    choice_loci2 = np.random.randint(self.n,size = n_mut2)
                    
                    if n_mut > self.N*I[-1][orig_strain,vacc]:
                        n_mut = int(I[-1][orig_strain,vacc])
                    if n_mut2 > Ntnew[orig_strain,vacc]:
                        n_mut2 = Ntnew[orig_strain,vacc]
                        #this should be very very unlikely
                        
                    for mutation in range(n_mut):
                        choice_strain = self.mut_from_to[orig_strain][choice_loci[mutation]]
                        
                        if self.B[choice_strain,vacc] == 1:
                            if sum(I[-1][choice_strain,:]) == 0.:
                                Ntnew[choice_strain][vacc] += 1.
                            else:
                                Inew[choice_strain][vacc] += 1./self.N
                            Inew[orig_strain][vacc] -= 1./self.N
                            
                    for mutation2 in range(n_mut2):
                        choice_strain = self.mut_from_to[orig_strain][choice_loci2[mutation2]]
                        
                        if self.B[choice_strain,vacc] == 1:
                            if sum(I[-1][choice_strain,:]) == 0.:
                                Ntnew[choice_strain][vacc] += 1.
                            else:
                                Inew[choice_strain][vacc] += 1./self.N
                            Ntnew[orig_strain][vacc] -= 1
            
            Inew[Inew <0] = 0. #this should never be an issue either.
            
            
            #micro poisson evolution
            for strain in range(len(self.strains)):
                allevents = deltat*(self.gamma +
                                    self.delta + 
                                    beta*np.sum(S[-1]*self.B[strain]))
                options = beta*S[-1]*self.B[strain]*deltat
                options = np.append(options,self.gamma*deltat)
                options = np.append(options,self.delta*deltat)
                options = options/sum(options)
                
                if Ntnew[strain,:].sum() == 0:
                    gill_time[strain] = Time[-1] + deltat
                    continue
                
                #GILLESPIE SSA
                if self.Xc > Ntnew[strain,:][np.nonzero(Ntnew[strain,:])].min() and Ntnew[strain,:].sum() >0:
                    if is_extinct[strain] == True:
                        gill_time[strain] += np.random.exponential(deltat/allevents/Ntnew[strain,:].sum())
                        is_extinct[strain] = False
                        self.tester += 1
                        print self.tester, Time[-1]
                        
                    while gill_time[strain] < Time[-1] +deltat and Ntnew[strain,:].sum() > 0:
                        choice_vaccine = np.random.choice(range(self.n+1),p=Ntnew[strain,:]/Ntnew[strain,:].sum())
                        choice = np.random.choice(range(self.n +3), p=options)
                        
                        if choice <= self.n:
                            Ntnew[strain,choice] += 1
                            Snew[choice] -= 1./self.N
                        if choice == self.n + 1:
                            Ntnew[strain,choice_vaccine] -= 1
                            Rnew[choice_vaccine] += 1./self.N
                        if choice == self.n + 2:
                            Ntnew[strain,choice_vaccine] -= 1
                            Dnew[strain] += 1./self.N
                            
                        if Ntnew[strain,:].sum() > 0:
                            gill_time[strain] += np.random.exponential(deltat/allevents/Ntnew[strain,:].sum())
                        else:
                            is_extinct[strain] = True
                            self.tester -= 1
                            print self.tester, Time[-1]
                    
                    
                #TAU LEAP
                else:
                    for vaccine in range(self.n + 1):
                        if Nt[-1][strain,vaccine] >0:
                            n_event = np.random.poisson(Nt[-1][strain,vaccine]*allevents)
                            #if n_event > 0:
                                #print n_event/Nt[-1][strain,vaccine]
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
                    gill_time[strain] = Time[-1] + deltat
                        
                    Ntnew[Ntnew <= 0] = 0.
                    Snew[Snew <= 0] = 0.
                  
            for strain in range(len(self.strains)):
                extinction_N1 = np.array(Ntnew[strain,:]).sum()
                extinction_N2 = np.array((Inew[strain,:]*self.N + Ntnew[strain,:])).sum()
            
                #FIXATION
                if extinction_N1 > self.from_stochastic:
                    self.e_time[strain] = Time[-1]
                    Inew[strain,:] += Ntnew[strain,:]/self.N
                    Ntnew[strain,:] = 0.
                
                #EXTINCTION
                if extinction_N2 < self.to_stochastic:
                    Ntnew[strain,:] += (Inew[strain,:]*self.N).astype(int)
                    Rnew[:] += (Inew[strain,:]*self.N \
                                - (Inew[strain,:]*self.N).astype(int))/self.N
                    Inew[strain,:] = 0.
                        
            end = time.time()
                
            thattime += end - start
            rep_num.append(beta/(self.gamma+self.delta)*(
                    np.matmul(self.B,Snew)))
            selection_coeff.append(((rep_num[-1]*(self.gamma+self.delta)-beta*Snew[0])/
                               (beta*Snew[0])))
            pfix.append(-np.log(1.-pfix[-1])/(1+selection_coeff[-1][1]))
            Nt.append(Ntnew)
            S.append(Snew)
            I.append(Inew)
            Itot.append(Inew + Ntnew/self.N)
            R.append(Rnew)
            D.append(Dnew)
            risk.append(1.-np.exp(-self.N*self.p*self.fixation_p(1.+selection_coeff[-1]
                   )*np.matmul(self.A,Itot[-1].sum(1))))
            cases.append(sum(sum(Itot[-1])) + sum(R[-1]) + sum(D[-1]))
            Time.append(Time[-1]+deltat)
            total.append(sum(sum(Inew + Ntnew/self.N))
                                + sum(Rnew + Snew) + sum(Dnew))
            if (Inew+Nt>0).sum() == 0:
                break
            
       # print np.array([thistime, thattime])/(thistime + thattime)
        #print 'total' , sum(sum(Inew + Ntnew/self.N)) + sum(Rnew + Snew) + sum(Dnew)
        self.total_dead = sum(D[-1])
        print is_extinct
        
        self.graphs = {'S':S,'Itot':Itot,'I':I,'R':R,'D':D,'cases':cases,
                'Time':Time,'Rt':rep_num,'te':emergetimes,'tl':losstimes,
                'Nt': Nt, 'risk':risk,'total':total,'selection_coeff':selection_coeff, 'pfix':pfix}
        return self.graphs
    
    #%%
    
    def plot_nice(self,title,limes):  
        fig, ax1 = plt.subplots()
        
        tv = self.vacctime
        th = tv + 1./self.theta*(1. - (self.gamma+self.delta)/self.beta3)
        color = 'tab:blue'
        ax1.set_xlabel('Time (Days)',size=20)
        ax1.set_ylabel('# Wildtype Infections', color=color,size=20)
        ax1.plot(self.graphs['Time'],np.array(self.graphs['Itot']).sum(2)[:,0]*self.N, color=color,label="W")
        ax1.tick_params(axis='y', labelcolor=color)
        plt.title(title,size=20)
        ax2 = ax1.twinx()
    
        for r in range(int(tv),int(th)):
            plt.axvline(x=r,color='y',alpha=0.01)
        color = 'tab:red'
        ax2.set_ylabel("# Mutant Infections", color=color,size=20)  # we already handled the x-label with ax1
        ax2.plot(self.graphs['Time'],np.array(self.graphs['Itot']).sum(2)[:,1]*self.N, color=color,label="M")
        ax2.tick_params(axis='y', labelcolor=color)
        if limes >0:
            ax2.set_ylim(0,limes)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        
        plt.axvline(x=th,color='y',alpha=0.5)
        
        plt.show()
        
    def plot_me(self,plot_lines):
        
        
        plt.plot(self.graphs['Time'],self.graphs['total'])
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('total',size = 20)
        plt.legend(range(self.n+1),loc='best')
        plt.show()
        
        plt.plot(self.graphs['Time'],self.graphs['S'])
        #plt.plot(self.graphs['Time'],1.4- np.array(self.graphs['Time'])*self.theta)
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
        #plt.axvline(x=th,color='y',alpha=0.5)
        #plt.axvline(x=vt,color='y',alpha=1.)
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
        
        plt.plot(self.graphs['Time'],np.array(self.graphs['Nt'])[:,0,0])
        plt.plot(self.graphs['Time'],np.array(self.graphs['Nt'])[:,1,0])
        plt.plot(self.graphs['Time'],np.array(self.graphs['Nt'])[:,0,1])
        plt.plot(self.graphs['Time'],np.array(self.graphs['Nt'])[:,1,1])
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('fractions Nt types',size = 20)
        #plt.legend(self.strains,loc='best')
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
        plt.axhline(y=1.,color='k',alpha=0.5)
        tv = self.vacctime
        th = tv + 1./self.theta*(1. - (self.gamma+self.delta)/self.beta3)
        plt.axvline(x=th,color='k',alpha=0.5)
        plt.axvline(x=tv,color='k',alpha=0.5)
        plt.legend(self.strains,loc='best')
        plt.show()

        plt.plot(self.graphs['Time'],self.graphs['D'])
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('fraction Dead',size = 20)
        plt.legend(self.strains,loc='best')
        plt.show()
        
        plt.plot(self.graphs['Time'],self.graphs['risk'])
        plt.plot(self.graphs['Time'],self.graphs['pfix'])
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('risk',size = 20)
        plt.legend(self.strains,loc='best')
        plt.show()
        
        plt.plot(self.graphs['Time'],self.graphs['selection_coeff'])
        plt.xlabel('timesteps',size = 20)
        plt.ylabel('selection coeff.',size = 20)
        plt.legend(self.strains,loc='best')
        plt.show()