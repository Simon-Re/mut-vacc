# mut-vacc: Mutations in the SIR Model with Vaccination

This repository contains the relevant code used to simulate the results in <https://www.medrxiv.org/content/10.1101/2021.02.08.21251383v1>.
3 Files can be found here. **mutvacc_tools.py**, which contains most of the functionality and the main class. The other two files, **generate_graphs.py** and **generate_2D.py**, contain utilities for running the main code multiple times and visualizing the results. While above publication deals with a single locus, the code is general enough to account for any number of targets and vaccines. 

## mutvacc_tools.py

Contains, aside some auxillary functions, the main class and the function **run-stochastic**. This function simulates a population in our extended SIR model for a time period of time T. For details on the model, please see <https://www.medrxiv.org/content/10.1101/2021.02.08.21251383v1>. All relevant parameters can be accessed and modified before a simulation is run. 

## generate_graphs.py 

Prduces graphs of the infection numbers of wildtype strains and emergent strain over the course of a simulation. Other quantities, such as the number of vaccinated individuals, the number of recovered individuals, etc. can be accessed and plotted via the **model.graphs** dictionary. 

## generate_2D.py

Produces graphs of macrovariables, such as the total death count at the end of a simulation or the appearance of an emergent strain after a given time. These macrovariables can be studied as a function of all parameters of the model. 
