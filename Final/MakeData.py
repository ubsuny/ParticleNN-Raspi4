from ExpodentialDistribution import Exp
from GaussianDistribution import Gauss
import numpy as np
import tensorflow as tf

def MakeData(p, Ntrain):
    '''Particle Neural Network: Generates Gaussian distribution masses and simulation Particle Data, to be used to either test or train the Particle Neural Network.'''
    
    params = 5
    
    ETrain,ETrainVals = Exp(p,Ntrain)

    GTrain,GTrainVals = Gauss(p,Ntrain)
    
    Train = np.zeros((2*Ntrain,params+1))

    Train[:Ntrain,0:params] = ETrain
    Train[Ntrain:,0:params] = GTrain
    Train[:Ntrain,params] = ETrainVals
    Train[Ntrain:,params] = GTrainVals

    BUFFER_SIZE = Ntrain*2
    BATCH_SIZE = Ntrain*2
    train_dataset = tf.data.Dataset.from_tensor_slices(Train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    for i in train_dataset.take(-1):
        train = i[0:,0:params]
        train_vals = i[0:,params]
        
    train = np.array(train)
    train_vals = np.array(train_vals)
        
    return train, train_vals