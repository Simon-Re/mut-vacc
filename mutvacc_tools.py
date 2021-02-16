import numpy as np
from itertools import product

def distance(l1,l2):
    """Computes the Levenshtein distance between two bitsrings.
    Args:
        l1 (str): bit sequence A 
        l2 (str): bit sequence B
    Returns:
        d (int): distance between sequence A and B
    """
    d = 0
    for i in range(len(l1)):
        if l1[i] is not l2[i]:
            d += 1
    return d

def vaccination_array(n):
    """Computes the 1-dim array of allowed vaccination rates of individuals with vaccine status i.
    Args:
        n (int): number of vaccines in the system
    Returns:
        C (numpy array, dim (n+1,1)): 1 if vaccination flow is positive, -1 if flow is negative
    """
    C = np.zeros(((n+1,1))) #vaccination matrix
    C[0][0] = -n - 1
    C = C +1
    return C

def mutation_and_resistance(strains,n):
    """Generates the structure of mutations and resistance between different strains and vaccines.
    Args:
        n (int): number of vaccines in the system
        strains (list of str, len = s): list of bitstrings in the system
    Returns:
        A (numpy array, dim (s,s)): 1 if mutation transition possible, 0 else
        B (numpy array, dim (s,n)): 1 if strain is resitant agains vaccine i
        mutation_dict (dict): A in form of a dictionary of neighbors.
    """
    
    A = np.zeros((len(strains),len(strains))) #mutation network
    B = np.zeros((len(strains),n)) #resistance matrix
   
    mutation_dict = {}
    
    for obj1 in list(enumerate(strains)):
        mutation_dict[obj1[0]] = []
        for obj2 in list(enumerate(strains)):
            if distance(obj1[1],obj2[1]) == 1:
                A[obj1[0],obj2[0]] = 1
                mutation_dict[obj1[0]].append(obj2[0])
        B[obj1[0]] = list(obj1[1])
        
    return A, B, mutation_dict

class mutvaccmodel:
# =============================================================================
#     The mutvaccmodel class
#
# Contains the following functions:
    #setup: initializes functions that are dependent on the number of vaccines n
    #beta_ld: returns the time dependent function of the the transmission beta
    #vaccspeed: returns the time dependent vaccination function
    #run_stochastic: runs the simulation
# =============================================================================
    
    def __init__(self):
        # =============================================================================
        # DEFAULT VALUES
        # =============================================================================
        
        #populationsize
        self.N = 10000
        
        #transmission rate before first phase of low transmission
        self.beta = 0.18
        
        #recovery rate
        self.gamma = 1./14
        
        #vaccination parameter theta_0
        self.theta = 0.002
        self.theta2 = self.theta
        
        #death rate
        self.delta = 0.01
        
        #probability of mutations per individual per day
        self.p = 0.00001
        
        #natural rate of immune loss
        self.mu = 1./180
        
        #start of vaccination compaign
        self.vacctime = 100
        
        #fraction of vaccine hesitants in the system
        self.hesitants = 0.01
        
        #number of wildtype infections at the beginning of the simulation
        self.small_fraction = 10./self.N
        
        #will there be phases of low trasmission?
        self.measures = True
        
        #thresholds of entering phase of low transmission or leaving the same phase
        self.lockdown_k = [0.01,0.001]
        
        #transmission rates may change after the first phase of low transmission was initiated
        self.beta2 = self.beta/3 #low transmission
        self.beta3 = self.beta*2/3. #high transmission
        
        #thresholds for treating the dynamics as deterministic mean field
        self.from_stochastic = 0.01
        self.to_stochastic = 0.1
        
        #saturation paramter of vaccination
        self.michaelis_menten = 0.01
        
        # Low transmission at the end of vaccination campaign when herd immunity is reached
        self.smart_lockdown = False #is such a "smart lockdown" implemented at all?
        self.smart_lockdown_T = 0 #length of this phase
        self.smart_lockdown_beta= self.beta3 #transmission rate in this phase
        
        # threshold for SSA GIllespie vs. Tau Leaping Algorithm
        self.Xc = 10
        
        
    def setup(self,n):
        #This function sets up the structure for mutation, resistance and vaccination among 
        #different vaccines and strains in the system
        
        genes = '01'
        strains = []
        for g in product(genes,repeat=n):
            strains.append("".join(g))
        
        C = vaccination_array(n)
        A, B, mutation_dict = mutation_and_resistance(strains,n)
        
        self.mut_from_to = mutation_dict
        self.A = A
        self.C= np.append(C,np.zeros((n+1,n)),1)
        self.M = A - n*np.identity(len(strains))
        self.B = np.append(np.array(np.ones((len(strains),1))),B,1)
        self.strains = strains
        self.n = n
        
    def beta_ld(self,infected,t):
        #This function takes the total number of infected individuals and computes the current 
        #transmission rate. it changes the class variables lockdown and openup
        #If smart_lockdown is activated, a period low transmission is activated around time th,
        #when approximately 60% are vaccinated (herdimmunity).
        
        
        #when vaccination is linear (low michaelis_menten and hesitants), then herdimmunity 
        #is reached at time th:
        th = self.vacctime + 1./self.theta*(1.-self.gamma/self.beta3) 
        
        if not self.measures: #if no transmission reducing measures are implemented 
            return self.beta
        
        if infected > self.lockdown_k[0] or self.lockdown: #entering low transmission
            self.lockdown = 1
            self.openup = 0
            b = self.beta2
        if not self.openup and not self.lockdown: #at the beginning of simulatoin
            b = self.beta
        if (self.lockdown and infected < self.lockdown_k[1]) or self.openup: #exiting low transmission
            self.openup = 1
            self.lockdown = 0
            b = self.beta3
        if th - int(self.smart_lockdown_T/2.) < t < th +int(
                self.smart_lockdown_T/2.) and self.smart_lockdown: #smart lockdown when herd immunity
            self.lockdown = 1
            self.openup = 0
            b = self.smart_lockdown_beta
        return b
    
    def vaccspeed(self,S0,R0,I0):
        #vaccination speed computed according to linear function with saturation. When many
        #people are alrady vaccinated the speed decreases.
        x = S0 + R0 + I0
        if x <= self.hesitants:
            return 0.
        y = S0 + R0 + self.michaelis_menten
        return (1.-self.hesitants/x)/y
        
    def run_stochastic(self,T, deltat):
        
        # =============================================================================
        # INITIALIZATION OF PARAMETERS
        # =============================================================================
        
        self.started_vacc = 0 #logical, records if vaccination campaign started
        self.lockdown = 0 #logical, records if in low transmission period
        self.openup = 0 #logical, records if in high transmission period
        
        total = [1.] #total number of individuals normalized by N
        
        D= [] #list of dead individuals in time
        
        R, S = [], []  
        #list of recovered, recovered vaccinated, susceptible and vaccinated individuals in time
        
        I, Nt,Itot = [], [], []
        #number of infected individual in small numbers and in deterministic regime
        
        cases = []
        #number of cases in the model, infected + recovered + dead
        
        rep_num = []
        #reproduction numbers as a function of time
        
        selection_coeff = []
        #selection coefficients V/S as a function of time
        
        Time = []
        #list of simulation times
        
        emergetimes = []
        losstimes = []
        is_extinct = []
        self.e_time = [0.]
        for l in range(len(self.strains)):
            emergetimes.append([])
            losstimes.append([])
            self.e_time.append(T)
            is_extinct.append(1) #logical, collects the state of existance
        emergetimes[0].append(0.)
        #collects times of virual emergence and loss
        
        I.append(np.zeros((len(self.strains), self.n+1)))
        Itot.append(np.zeros((len(self.strains), self.n+1)))
        Nt.append(np.zeros((len(self.strains), self.n+1)))
        S.append(np.zeros((self.n+1)))
        R.append(np.zeros((self.n+1)))
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
        Time.append(0)
        #initialize first elements of the above lists 
        
        gill_time = np.zeros(len(self.strains))
        #Current time of the Gillespie SSA reaction scheme, that will be computed
        #with exponentially distributed timings
        
        #START OF THE SIMULATION / MAIN LOOP
        for t in range(T-1):
            
            # =============================================================================
            # DETERMINISTIC DYNAMICS evaluated with an EULER FORWARD INTEGRATION
            # =============================================================================
            
            beta = self.beta_ld(np.sum(Itot[-1]),Time[-1])
            
            Inew = I[-1] + S[-1]*(I[-1].sum(1)*self.B.T).T*beta*deltat \
                    - I[-1]*self.gamma*deltat - I[-1]*self.delta*deltat 
                    
            Dnew = D[-1] + I[-1].sum(1)*self.delta*deltat
            
            if self.started_vacc or Time[-1] > self.vacctime: #condition for vaccination start
                
                self.started_vacc = 1
                v_vacc = self.vaccspeed(S[-1][0],R[-1][0],sum(Itot[-1][:,0]))
                
                Rnew = R[-1] - self.mu*R[-1]*deltat + sum(I[-1])*self.gamma*deltat \
                        + self.theta2/self.n*self.C.T[0]*deltat*v_vacc*R[-1][0]
                        
                Snew = S[-1] + self.mu*R[-1]*deltat \
                        - S[-1]*np.matmul(np.transpose(self.B),
                           np.sum(I[-1],1))*beta*deltat \
                        + self.theta/self.n*self.C.T[0]*deltat*v_vacc*S[-1][0]
            else:
                Rnew = R[-1] - self.mu*R[-1]*deltat + sum(I[-1])*self.gamma*deltat
                Snew = S[-1] + self.mu*R[-1]*deltat \
                    - S[-1]*np.matmul(np.transpose(self.B),
                       np.sum(I[-1],1))*beta*deltat
            
            Inew[Inew < 0] = 0. # In case the integrator overshoots
            
            Ntnew = np.copy(Nt[-1])
            
            # =============================================================================
            # MUTATIONS ARE COMPUTED AS POISSON RANDOM VARIABLES for the number of infected
            # individuals in the deterministic regime Nt
            # =============================================================================
            
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
                        #this should be very unlikely
                        
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
            
            Inew[Inew <0] = 0.
            
            # =============================================================================
            # STOCHASTIC SIMULATION FOR SMALL POPULATION SIZES
            # =============================================================================
 
            for strain in range(len(self.strains)):
                #DERIVATION OF PROPENSITY FUNCTIONS
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
                
                # =============================================================================
                # GILLESPIE SSA ALGORITHM if Nt < Xc, Reaction times are drawn from an exponential 
                # distribution and reactions are chosen according to their propensity functions
                # =============================================================================
                
                if self.Xc > Ntnew[strain,:][np.nonzero(Ntnew[strain,:])].min() and Ntnew[strain,:].sum() >0:
                    if is_extinct[strain] == True:
                        gill_time[strain] += np.random.exponential(deltat/allevents/Ntnew[strain,:].sum())
                        is_extinct[strain] = False
                        
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
                            #a new reactiontime is drawn from an exponential distribution
                            gill_time[strain] += np.random.exponential(deltat/allevents/Ntnew[strain,:].sum())
                        else:
                            is_extinct[strain] = True

                # =============================================================================
                # TAU LEAP ALGORITHM if Nt > Xc , the number of effective reactions during a timestep
                # is drawn from a Poisson Distribution
                # =============================================================================
                
                else:
                    for vaccine in range(self.n + 1):
                        if Nt[-1][strain,vaccine] >0:
                            #number of events drawn from Poisson
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
                    gill_time[strain] = Time[-1] + deltat
                        
                    Ntnew[Ntnew <= 0] = 0.
                    Snew[Snew <= 0] = 0.
                    
            
            # =============================================================================
            # CHAINGING FROM STOCHASTIC REGIME TO DETERMINISTIC REGIME, and reverse
            # =============================================================================
            
            for strain in range(len(self.strains)):
                extinction_N1 = np.array(Ntnew[strain,:]).sum()
                extinction_N2 = np.array((Inew[strain,:]*self.N + Ntnew[strain,:])).sum()
            
                #TO THE DETERMINISTIC REGIME 
                if extinction_N1 > self.from_stochastic:
                    self.e_time[strain] = Time[-1]
                    Inew[strain,:] += Ntnew[strain,:]/self.N
                    Ntnew[strain,:] = 0.
                
                #FROM THE DETERMINISTIC REGIME
                if extinction_N2 < self.to_stochastic:
                    Ntnew[strain,:] += (Inew[strain,:]*self.N).astype(int)
                    Rnew[:] += (Inew[strain,:]*self.N \
                                - (Inew[strain,:]*self.N).astype(int))/self.N
                    Inew[strain,:] = 0.
                        

            # =============================================================================
            # RECORDING UPDATED VARIABLES IN self.graphs
            # =============================================================================
            
            rep_num.append(beta/(self.gamma+self.delta)*(
                    np.matmul(self.B,Snew)))
            selection_coeff.append(((rep_num[-1]*(self.gamma+self.delta)-beta*Snew[0])/
                               (beta*Snew[0])))
            Nt.append(Ntnew)
            S.append(Snew)
            I.append(Inew)
            Itot.append(Inew + Ntnew/self.N)
            R.append(Rnew)
            D.append(Dnew)
            cases.append(sum(sum(Itot[-1])) + sum(R[-1]) + sum(D[-1]))
            Time.append(Time[-1]+deltat)
            total.append(sum(sum(Inew + Ntnew/self.N))
                                + sum(Rnew + Snew) + sum(Dnew))
            if (Inew+Nt>0).sum() == 0:
                break
            
       # print np.array([thistime, thattime])/(thistime + thattime)
        #print 'total' , sum(sum(Inew + Ntnew/self.N)) + sum(Rnew + Snew) + sum(Dnew)
        self.total_dead = sum(D[-1])
        #print is_extinct
        
        self.graphs = {'S':S,'Itot':Itot,'I':I,'R':R,'D':D,'cases':cases,
                'Time':Time,'Rt':rep_num,
                'Nt': Nt, 'total':total,'selection_coeff':selection_coeff}
        return self.graphs