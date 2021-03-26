import numpy as np
import random
import math

def Exp(p, Ntrain):
    '''Particle Neural Network: Generates Expodential distribution masses and simulation Particle Data, to be used to either test or train the Particle Neural Network.'''
    
    MassE = np.random.exponential(p,Ntrain)
    cos = np.vectorize(math.cos)
    acos = np.vectorize(math.acos)
    sin = np.vectorize(math.sin)
    sqrt = np.vectorize(math.sqrt)
    theta = np.random.uniform(0,2*math.pi,Ntrain)
    #phi = np.random.uniform(0,math.pi,Ntrain)
    phi = acos(sin(theta))
    PxE = p*(cos(theta))*(cos(phi))
    PyE = p*(cos(theta))*(sin(phi))
    PzE = p*(cos(phi))

    TrainValues = np.zeros(Ntrain)
    TrainValues = TrainValues.astype(int)

    Train = [[]]
    for k in range(Ntrain):
        Train.append(np.array([PxE[k],PyE[k],PzE[k],MassE[k]]))
        
    Train.pop(0)
    Train = np.asarray(Train)
        
    return Train, TrainValues


    