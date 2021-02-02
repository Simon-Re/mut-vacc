import numpy as np
import matplotlib.pyplot as plt
import mutvacc_tools as mv
import os
import time

def get_task_params(task_id,n1,n2,n3):
    #n1*n2*n3 = len(task_id)
    i1 = np.remainder(task_id,n1)
    i2 = np.remainder(int(task_id/n1),n2)
    i3 = np.remainder(int(task_id/(n1*n2)),n3)
    return i1,i2,i3

def save_pic(i):
    plt.savefig(experiment_title+ '/fig_' + str(i))
    i += 1
    return i
    
experiment_title = 'smart_lockdown_study'

if os.path.exists(experiment_title):
    print 'does already exist, proceed?'
    raw_input()
if not os.path.exists(experiment_title):
    os.makedirs(experiment_title)
   
n = 1
model = mv.mutvaccmodel()
model.setup(n)

deltat = 1.
T= int(365.*3/deltat)
Imeas = 1.
model.lockdown_k = [Imeas,Imeas]
model.beta = 0.18
model.delta = 1./14.*0.01
model.gamma = 1./14.*0.99
model.beta2 = 0.055
model.beta3 = model.beta
theta = 0.001
model.theta = theta
model.theta2 = theta
model.vacctime = 365
model.k = 1.
model.mu = 1./180
model.p = 1e-4
th = model.vacctime + 1./model.theta*(1.-model.gamma/model.beta3)
model.N = 100000000
model.ext_value = 0.1
model.hesitants = 0.0
model.from_stochastic = 1000
model.to_stochastic = 500
model.small_fraction = 100./model.N
model.smart_lockdown = False
model.Xc = 200
model.michaelis_menten = 0.00001

ax1 = 50
ft1 = [0.01,0.07]
ax1space = np.linspace(ft1[0],ft1[1],ax1)
tit1 = r'$\beta$'

ax2 = 15
ft2 = [15,200]
tit2 = r'$T$'
ax2space = np.linspace(ft2[0],ft2[1],ax2)

tit3 = r'p'

iterations = 30


f = open(experiment_title+"/parameters.txt", "w")
for entry in model.__dict__:
    if entry in ['theta2','theta', 'hesitants', 
                 'k','m','N','mu','p','beta','delta',
                 'small_fraction','n','gamma']:
        f.write(entry +' ' + str(model.__dict__[entry]) + '\n')
f.write('T ' + str(T) + '\n')
f.write('deltat ' + str(deltat))
f.close()

pic = 0

for prob in [0]:
    mat = np.zeros((n+1,ax2,ax1))
    mat2 = np.zeros((n+1,ax2,ax1))
    one, two = 0, 0
    
    #%%
    
    for pressure in ax1space:
        one = 0
        for ld_time in ax2space:
            model.vacctime = 500
            T = int(ld_time)
            model.beta = pressure
            model.p = prob
            for i in range(iterations):
                print T, pressure, 'here'
                start = time.time()
                
                model.run_stochastic(T,deltat)
                
                end = time.time()
                print('absolute', end - start)

                mat2[:,one,two] += np.heaviside(model.graphs['Itot'][-1].sum(1),0)/float(iterations)
                mat[:,one,two] += model.graphs['D'][-1]/iterations
            #model.plot_me(False)
            one += 1
        two += 1
    
    #%%
    
    plt.imshow(mat[0], cmap=plt.cm.Greys, interpolation='none', 
               extent=[ft1[0],ft1[1],ft2[1],ft2[0]], aspect="auto")
    plt.ylabel(tit2,size=20)
    plt.xlabel(tit1,size=20)
    plt.title('Wildtype Deaths in ' + str(iterations)+ ' runs \n' + 
              tit3 +' = ' +str(prob),size =15)
    plt.colorbar()
    pic = save_pic(pic)
    plt.show()
    
    
    plt.imshow(mat[1], cmap=plt.cm.Greys, interpolation='none', 
               extent=[ft1[0],ft1[1],ft2[1],ft2[0]], aspect="auto")
    plt.ylabel(tit2,size=20)
    plt.xlabel(tit1,size=20)
    plt.title('Mutant Deaths in ' + str(iterations)+ ' runs \n' + 
              tit3 + ' = ' +str(prob),size =15)
    plt.colorbar()
    pic = save_pic(pic)
    plt.show()
    
    
    plt.imshow(mat[1]+mat[0], cmap=plt.cm.Greys, interpolation='none', 
               extent=[ft1[0],ft1[1],ft2[1],ft2[0]], aspect="auto")
    plt.ylabel(tit2,size=20)
    plt.xlabel(tit1,size=20)
    plt.title('All Deaths in ' + str(iterations)+ ' runs \n' + 
              tit3+ ' = ' +str(prob),size =15)
    plt.colorbar()
    pic = save_pic(pic)
    plt.show()
    
    #%%
    plt.imshow(mat2[1], cmap=plt.cm.Greys, interpolation='none', 
               extent=[ft1[0],ft1[1],ft2[1],ft2[0]], aspect="auto")
    plt.ylabel(tit2,size=20)
    plt.xlabel(tit1,size=20)
    plt.title('Mutant Emergence in ' + str(iterations)+ ' runs \n' + 
              tit3 + ' = ' +str(prob),size =15)
    plt.colorbar()
    pic = save_pic(pic)
    plt.show()
    
    plt.imshow(mat2[0], cmap=plt.cm.Greys, interpolation='none', 
               extent=[ft1[0],ft1[1],ft2[1],ft2[0]], aspect="auto")
    plt.ylabel(tit2,size=20)
    plt.xlabel(tit1,size=20)
    plt.title('Strain Survival in ' + str(iterations)+ ' runs \n' + 
              tit3 + ' = ' +str(prob),size =15)
    plt.colorbar()
    pic = save_pic(pic)
    plt.show()
    
    #%%
    
    N = 100# model.small_fraction*model.N
    x = np.arange(0.01, 0.07, 0.001)
    
    y = np.arange(15, 200, 1)
    
    xx, yy = np.meshgrid(x, y)
    
    z = 1 -((1./14*np.exp((1./14 - xx)*yy) - 1./14.)/(1./14*np.exp((1./14 - xx)*yy) - xx))**N
    
    h = plt.contourf(x,y,z, cmap=plt.cm.Greys)
    plt.colorbar()
    plt.title(r'$1- \left( \frac{(\delta +\gamma)e^{(\delta + \gamma - \beta)T} - \gamma - \delta}{(\delta +\gamma)e^{(\delta + \gamma - \beta)T}-\beta} \right)^{I_0}}$', size = 20, pad=20)
    plt.xlabel(r'$\beta$', size = 20)
    plt.ylabel(r'$T$', size = 20)
    plt.gca().invert_yaxis()
    plt.show()