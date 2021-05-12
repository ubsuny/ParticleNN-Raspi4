'''expodential_distribution.py generates the data to
be used for one classification for all neural networks.'''
import math
import numpy as np

def exp(momentum, n_train):
    '''Particle Neural Network: Generates Expodential distribution
    masses and simulation Particle Data, to be used to either test
    or train the Particle Neural Network.'''
    mass_exp = np.random.exponential(momentum,n_train)
    cos = np.vectorize(math.cos)
    sin = np.vectorize(math.sin)
    phi = np.random.uniform(0,2*math.pi,n_train)
    theta = np.random.uniform(-math.pi,math.pi,n_train)
    px_exp = momentum*(cos(phi))*(sin(theta))
    py_exp = momentum*(sin(theta))*(sin(phi))
    pz_exp = momentum*(cos(theta))
    tau_exp = np.random.exponential(momentum*1.5,n_train)
    train_values = np.zeros(n_train)
    train_values = train_values.astype(int)
    train = [[]]
    for k in range(n_train):
        train.append(np.array([px_exp[k],py_exp[k],pz_exp[k],mass_exp[k],tau_exp[k]]))
    train.pop(0)
    train = np.asarray(train)
    return train, train_values
