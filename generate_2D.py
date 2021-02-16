import numpy as np
import matplotlib.pyplot as plt
import mutvacc_tools as mv

# =============================================================================
# Initialization of the Model Paramters
# =============================================================================
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
model.p = 1e-6
model.N = 100000000
model.hesitants = 0.0
model.from_stochastic = 1000
model.to_stochastic = 900
model.small_fraction = 100./model.N
model.smart_lockdown = False
model.Xc = 100
model.michaelis_menten = 0.01

# =============================================================================
# SETTING OF PARAMETER SPACE TO EXPLORE, ADAPT FOR YOUR OWN QUESTIONS!
# The example explores the length of the simulation T vs. beta vs. p in a system
# with a small load of 100 initial infections.
# =============================================================================

ax1 = 10
ft1 = [0.01,0.07]
ax1space = np.linspace(ft1[0],ft1[1],ax1)
tit1 = r'$\beta$'

ax2 = 10
ft2 = [15,200]
tit2 = r'$T$'
ax2space = np.linspace(ft2[0],ft2[1],ax2)

tit3 = r'p'
ax3space = np.logspace(-6,-8,3)

iterations = 1 #number of iterations per bin

# =============================================================================
# RUN SIMULATION A COUPLE OF TIMES TO GENERATE 2D Contour Plot
# =============================================================================

pic = 0

for prob in ax3space:
    mat = np.zeros((n+1,ax2,ax1))
    mat2 = np.zeros((n+1,ax2,ax1))
    one, two = 0, 0
    
    for a in ax1space:
        one = 0
        for b in ax2space:
            
            model.vacctime = 500
            T = int(b)
            model.beta = a
            model.p = prob
            for i in range(iterations):
                model.run_stochastic(T,deltat) #run model
                
                #frequency of mutant emergence, e.g. is a strain still in the system
                #at the end of a simulation?
                mat2[:,one,two] += np.heaviside(model.graphs['Itot'][-1].sum(1),0)/float(iterations)
                
                #death count
                mat[:,one,two] += model.graphs['D'][-1]/iterations
            one += 1
        two += 1

    
    # =============================================================================
    # VISUALIZE RESULTS
    # =============================================================================
    
    plt.imshow(mat[0], cmap=plt.cm.Greys, interpolation='none', 
               extent=[ft1[0],ft1[1],ft2[1],ft2[0]], aspect="auto")
    plt.ylabel(tit2,size=20)
    plt.xlabel(tit1,size=20)
    plt.title('Wildtype Deaths in ' + str(iterations)+ ' runs \n' + 
              tit3 +' = ' +str(prob),size =15)
    plt.colorbar()
    plt.show()
    
    
    plt.imshow(mat[1], cmap=plt.cm.Greys, interpolation='none', 
               extent=[ft1[0],ft1[1],ft2[1],ft2[0]], aspect="auto")
    plt.ylabel(tit2,size=20)
    plt.xlabel(tit1,size=20)
    plt.title('Mutant Deaths in ' + str(iterations)+ ' runs \n' + 
              tit3 + ' = ' +str(prob),size =15)
    plt.colorbar()
    plt.show()
    
