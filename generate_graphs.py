import numpy as np
import matplotlib.pyplot as plt
import mutvacc_tools as mv

#LOAD A MUTVACC_TOOL CLASS AND INITIALIZING IT WITH A SINGLE STRAIN n=1
n = 1
model = mv.mutvaccmodel()
model.setup(n)



#SETTIG THE LENGTH T OF A SIMULATION, AS WELL AS THE EULER INTEGRATION STEP deltat
deltat = 0.2
T= int(365.*3/deltat)




#SETTING THE PARAMETERS OF THE SIMULATION RUN
Imeas = 0.015*0.1 #F, Threshold for switching into low transmission phase
model.lockdown_k = [Imeas,0.01*0.01] #Setting the upper bound F and lower bound.
model.beta = 0.18 #Transmission Rate at the beginning of the simulation
model.delta = 1./14.*0.01 #Death Rate
model.gamma = 1./14.*0.99 #Recovery Rate
model.beta2 = 0.055 #Transmission Rate during periods of low transmission
model.beta3 = model.beta #Transmission Rate after an initial period of low transmission
theta = 1./365. #vaccination speed, theta_0
model.theta = theta #for the susceptible population S
model.theta2 = theta #for the recovered population R
model.vacctime = 365 #the time at which vaccination starts, e.g. 1 yr
model.mu = 1./180 #rate of immunity loss for recovered individuals
model.p = 1e-6 #mutation probability p. individual p. day
model.N = 10000000 #total population size
model.hesitants = 0.0 #fraction of vaccine hesitants h in the system
model.from_stochastic = 1000 #Value N* to switch from Tau Leap to Euler Forward
model.to_stochastic = 900 #to switch from Euler Forward to Tau Leap
model.small_fraction = 10./model.N #Initial load of infected individuals
model.smart_lockdown = False #a phase of low transmission at the point of reaching herd immunity?
model.smart_lockdown_beta = model.beta #what is the transmission during a smart lockdown
model.smart_lockdown_T= 0 #how long does a smart lockdown last?
model.Xc = 100 #Value Xc to switch from Gillespie SSA to Tau Leap
model.michaelis_menten = 0.01 #Saturation coefficient k for vaccination




#RUN A SIMULATION FEATURING DETERMINISTIC AND STOCHASTIC DYNAMICS
model.run_stochastic(T,deltat)




#GENERATE PLOTS OF THE INFECTED POPULATION AS A FUNCTION OF TIME, as in FIG.1 
#(WORKS FOR n=1, adapt for n>1)
title = "# INFECTED"
limes = 0 #sets the scale of the plot for the mutant population
fig, ax1 = plt.subplots()

tv = model.vacctime #start of vaccination 
th = tv + 1./model.theta #end of vaccination
color = 'tab:blue'
ax1.set_xlabel('Time (Days)',size=20)
ax1.set_ylabel(r'$I_{wt}$', color=color,size=20)
ax1.plot(model.graphs['Time'],np.array(model.graphs['Itot']).sum(2)[:,0]*model.N, color=color,label="W")
ax1.tick_params(axis='y', labelcolor=color)
plt.title(title,size=20, loc='left')
ax2 = ax1.twinx()

for r in range(int(tv),int(th)):
    plt.axvline(x=r,color='y',alpha=0.01)
color = 'tab:red'
ax2.set_ylabel(r'$I_{r}+I_r^V$', color=color,size=20)
ax2.plot(model.graphs['Time'],np.array(model.graphs['Itot']).sum(2)[:,1]*model.N, color=color,label="M")
ax2.tick_params(axis='y', labelcolor=color)
if limes >0:
    ax2.set_ylim(0,limes)

plt.axvline(x=th,color='y',alpha=0.5)
plt.show()


