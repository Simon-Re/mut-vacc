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
    
experiment_title = 'test'

if os.path.exists(experiment_title):
    print 'does already exist, proceed?'
    raw_input()
if not os.path.exists(experiment_title):
    os.makedirs(experiment_title)
   
n = 1
model = mv.mutvaccmodel()
model.setup(n)
deltat =1

T= int(365.*3./deltat)
model.lockdown_k = [0.00014,0.00005]
model.beta = 0.22
model.beta2 = model.beta/3.
model.beta3 = model.beta*2.5/3.
model.theta = 0.001
model.theta2 = 0.001
model.vacctime = 365*2
model.k = 1.
model.mu = 1./180
model.p = 0.000001*deltat
model.N = 10000000
model.ext_value = 0.1
model.hesitants = 0.01
model.small_fraction = 10./model.N

ax1 = 20
ft1 = [0.00175,0.00225]
tit1 = r'$I_{measures}$'
ax2 = 20
ft2 = [0,800]
tit2 = r'$t_{vacc}$'
tit3 = r'p'
iterations = 10


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

for prob in [0.0005,0.0001,0.00001]:
    mat = np.zeros((n+1,ax2,ax1))
    mat2 = np.zeros((n+1,ax2,ax1))
    mat3 = np.zeros((n+1,ax2,ax1))
    one, two = 0, 0
    
    #%%
    
    for ks in np.linspace(ft1[0],ft1[1],ax1):
        one = 0
        for tv in np.linspace(ft2[0],ft2[1],ax2):
            model.lockdown_k[0] = ks
            model.vacctime = tv
            model.p = prob
            for i in range(iterations):
                print ks, tv, i
                start = time.time()
                
                model.run_stochastic(T,deltat)
                
                end = time.time()
                print('absolute', end - start)
                #model.plot_me()
                for l in range(len(model.strains)):
                    if len(model.graphs['te'][l]) >0:
                        mat3[l,one,two] += model.graphs['te'][l][0]
                mat2[:,one,two] += np.heaviside(model.graphs['D'][-1],0)/iterations
                mat[:,one,two] += model.graphs['D'][-1]/iterations
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
    
    plt.imshow(mat3[1]/mat2[1]/iterations, cmap=plt.cm.Greys, interpolation='none', 
               extent=[ft1[0],ft1[1],ft2[1],ft2[0]], aspect="auto")
    plt.ylabel(tit2,size=20)
    plt.xlabel(tit1,size=20)
    plt.title('Av. Mutant Emergence Time in ' + str(iterations)+ ' runs \n' + 
              tit3 + ' = ' +str(prob),size =15)
    plt.colorbar()
    pic = save_pic(pic)
    plt.show()