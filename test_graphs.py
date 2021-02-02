import numpy as np
import matplotlib.pyplot as plt
import mutvacc_tools as mv


n = 1
model = mv.mutvaccmodel()
model.setup(n)
#%%

deltat = 1.
T= int(365.*3/deltat)
Imeas = 0.015*0.1
model.lockdown_k = [Imeas,0.01*0.02]
model.beta = 0.18
model.delta = 1./14.*0.01
model.gamma = 1./14.*0.99
model.beta2 = 0.055
model.beta3 = model.beta
theta = 1./365.
model.theta = theta
model.theta2 = theta
model.vacctime = 365
model.k = 1.
model.mu = 1./180
model.p = 1e-6
th = model.vacctime + 1./model.theta*(1.-model.gamma/model.beta3)
model.N = 10000000
model.ext_value = 0.1
model.hesitants = 0.0
model.from_stochastic = 100
model.to_stochastic = 50
model.small_fraction = 10./model.N
model.smart_lockdown = False
model.Xc = 10
model.michaelis_menten = 0.01

model.run_stochastic(T,deltat)

        
herd_time = model.vacctime + int(1./model.theta*(1.- model.gamma/model.beta3))
if herd_time < len(model.graphs['Itot']):
    print herd_time, model.graphs['Itot'][herd_time][0,0]
else:
    print herd_time, model.graphs['Itot'][-1][0,0]
                           

#%%
    
model.plot_me(False)
title = 'p = 1e-6'
limes = 100
model.plot_nice(title,limes)

