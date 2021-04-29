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
    phi = np.random.uniform(0,2*math.pi,Ntrain)
    theta = np.random.uniform(-math.pi,math.pi,Ntrain)
    PxE = p*(cos(phi))*(sin(theta))
    PyE = p*(sin(theta))*(sin(phi))
    PzE = p*(cos(theta))
    tau = np.random.exponential(p*1.5,Ntrain)

    TrainValues = np.zeros(Ntrain)
    TrainValues = TrainValues.astype(int)

    Train = [[]]
    for k in range(Ntrain):
        Train.append(np.array([PxE[k],PyE[k],PzE[k],MassE[k],tau[k]]))
        
    Train.pop(0)
    Train = np.asarray(Train)
        
    return Train, TrainValues


    