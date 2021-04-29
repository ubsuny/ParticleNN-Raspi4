import numpy as np
import random
import math

def Gauss(p, Ntrain):
    '''Particle Neural Network: Generates Gaussian distribution masses and simulation Particle Data, to be used to either test or train the Particle Neural Network.'''
    
    MassN = np.random.normal(p/2,math.sqrt(p),Ntrain)
    cos = np.vectorize(math.cos)
    acos = np.vectorize(math.acos)
    sin = np.vectorize(math.sin)
    sqrt = np.vectorize(math.sqrt)
    phi = np.random.uniform(0,2*math.pi,Ntrain)
    theta = np.random.uniform(-math.pi,math.pi,Ntrain)
    PxN = p*(cos(phi))*(sin(theta))
    PyN = p*(sin(theta))*(sin(phi))
    PzN = p*(cos(theta))

    TrainValues = np.ones(Ntrain)
    TrainValues = TrainValues.astype(int)

    Train = [[]]
    for k in range(Ntrain):
        Train.append(np.array([PxN[k],PyN[k],PzN[k],MassN[k]]))
        
    Train.pop(0)
    Train = np.asarray(Train)
        
    return Train, TrainValues