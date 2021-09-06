#### sosPy
### Based on the Journal Articles and Code by
##
## Min-Yuan Cheng, Doddy Prayogo,         
## Symbiotic Organisms Search: A new metaheuristic optimization algorithm, 
## Computers & Structures 139 (2014), 98-112   
## http://dx.doi.org/10.1016/j.compstruc.2014.03.007   
##
### Original Code by Doddy Prayogo - http://140.118.5.112:85/SOS/
### Written by Danny Gho at 
### National Taiwan University of Science and Technology (NTUST), Taipei, Taiwan

import numpy as np
import random

def objFun(x):
    result = 10*2 + (x[0]**2-10*np.cos(2*np.pi*x[0]))+ (x[1]**2-10*np.cos(2*np.pi*x[1]))
    return result

maxFE = 50000
ecoSize = 100

ub = np.array([5.12,5.12],dtype=np.float32)
lb = np.array([-5.12,-5.12],dtype=np.float32)
nvar = len(ub)

ecosystem = np.random.uniform(lb,ub, [ecoSize,nvar])
fitness = np.apply_along_axis(objFun, axis=1, arr=ecosystem)

best_organism =  np.zeros([1, nvar])
best_organism_id = 0

FE = 0 
while (FE < maxFE):
    for index in range(0,ecoSize):
        best_fitness, best_fitness_id = np.amin(fitness), np.argmin(fitness)
        best_organism = ecosystem[best_fitness_id,:]

        ## Mutualism Phase
        partner_idx = random.sample(set(range(0,ecoSize))-set([index]),1)
        partner = ecosystem[partner_idx,:]
        
        mutualVector = np.mean([ecosystem[index,:], partner])
        BF1 = np.round(1+np.random.uniform(0,1,1))
        BF2 = np.round(1+np.random.uniform(0,1,1))

        ecoNew1 = ecosystem[index,:] + np.multiply(np.random.uniform(size=[1,nvar]),(best_organism - (BF1 * mutualVector)))
        ecoNew2 = partner  + np.multiply(np.random.uniform(size=[1,nvar]),(best_organism - (BF2 * mutualVector)))
        
        ecoNew1 = np.clip(ecoNew1, lb, ub)[0]
        ecoNew2 = np.clip(ecoNew2, lb, ub)[0]
        
        fitnessNew1 = objFun(ecoNew1)
        fitnessNew2 = objFun(ecoNew2)

        if fitnessNew1 < fitness[index]:
            fitness[index] = fitnessNew1
            ecosystem[index,:] = ecoNew1

        if fitnessNew2 < fitness[partner_idx]:
            fitness[partner_idx] = fitnessNew2
            ecosystem[partner_idx,:] = ecoNew2

        if (False == (fitness == (np.apply_along_axis(objFun, axis=1, arr=ecosystem)))).any():
            print("Error!")

        FE = FE + 2

        ## Commensialism Phase
        partner_idx = random.sample(set(range(0,ecoSize))-set([index]),1)
        partner = ecosystem[partner_idx,:]

        ecoNew1 = ecosystem[index,:] + np.multiply((np.random.uniform(size=[1,nvar])*2-1),(best_organism-partner))
        ecoNew1 = np.clip(ecoNew1, lb, ub)[0]

        fitnessNew1 = objFun(ecoNew1)

        if fitnessNew1 < fitness[index]:
            fitness[index] = fitnessNew1
            ecosystem[index,:] = ecoNew1
        
        if (False == (fitness == (np.apply_along_axis(objFun, axis=1, arr=ecosystem)))).any():
            print("Error!")


        FE = FE + 1

        ## Parasitsm Phase
        partner_idx = random.sample(set(range(0,ecoSize))-set([index]),1)
        partner = ecosystem[partner_idx,:]
        if (False == (fitness == (np.apply_along_axis(objFun, axis=1, arr=ecosystem)))).any():
            print("Error!")

        parasite = ecosystem[index,:].copy()
        parasite_dim = random.sample(list(range(0,nvar)),max(int(np.random.uniform(0,nvar)),1))
        parasite[parasite_dim] = np.random.uniform(lb[parasite_dim],ub[parasite_dim], len(parasite_dim))
        fitnessParasite = objFun(parasite)
        if (False == (fitness == (np.apply_along_axis(objFun, axis=1, arr=ecosystem)))).any():
            print("Error!")

        if fitnessParasite < fitness[partner_idx]:
            fitness[partner_idx] = fitnessParasite
            ecosystem[partner_idx,:] = parasite
        
        if (False == (fitness == (np.apply_along_axis(objFun, axis=1, arr=ecosystem)))).any():
            print("Error!")

        FE = FE + 1 

best_fitness, best_fitness_id = np.amin(fitness), np.argmin(fitness)
best_organism = ecosystem[best_fitness_id,:]

print("Best Fitness {}".format(best_fitness))
print("Parameters {}".format(best_organism))
