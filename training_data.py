import tensorflow as tf
import numpy as np
from ExpodentialDistribution import Exp
from GaussianDistribution import Gauss

def train_data(p,Ntrain):
    params = 4
    
    ETrain,ETrainVals = Exp(p,Ntrain)

    GTrain,GTrainVals = Gauss(p,Ntrain)

    Train = np.zeros((2*Ntrain,params))
    Train_vals = np.zeros((2*Ntrain))

    Train[:Ntrain,:] = ETrain
    Train[Ntrain:,:] = GTrain
    Train_vals[:Ntrain] = ETrainVals
    Train_vals[Ntrain:] = GTrainVals
    
    return Train

