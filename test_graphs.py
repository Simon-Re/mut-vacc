import numpy as np
import matplotlib.pyplot as plt
import mutvacc_tools as mv


n = 1
model = mv.mutvaccmodel()
model.setup(n)
#%%

deltat = 1.
T= int(365.*3./deltat)
model.lockdown_k = [0.0014,0.0005]
model.beta = 0.22
model.beta2 = model.beta/3.
model.beta3 = model.beta*2.5/3.
model.theta = 0.01
model.theta2 = 0.01
model.vacctime = 365
model.k = 1.
model.mu = 1./180
model.p = 0.00001*deltat
model.N = 10000000
model.ext_value = 0.1
model.hesitants = 0.01
model.small_fraction = 10./model.N

model.run_stochastic(T,deltat)

model.plot_me(False)