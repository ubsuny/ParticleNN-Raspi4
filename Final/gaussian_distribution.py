'''gaussian_distribution.py generates the data to
be used for one classification for all neural networks.'''
import math
import numpy as np

def gauss(momentum, n_train):
    '''Particle Neural Network: Generates Gaussian distribution
    masses and simulation Particle Data, to be used to either test
    or train the Particle Neural Network.'''
    mass_gauss = np.random.normal(momentum/2,math.sqrt(momentum),n_train)
    cos = np.vectorize(math.cos)
    sin = np.vectorize(math.sin)
    phi = np.random.uniform(0,2*math.pi,n_train)
    theta = np.random.uniform(-math.pi,math.pi,n_train)
    px_gauss = momentum*(cos(phi))*(sin(theta))
    py_gauss = momentum*(sin(theta))*(sin(phi))
    pz_gauss = momentum*(cos(theta))
    tau_gauss = np.random.exponential(momentum*3,n_train)
    train_values = np.ones(n_train)
    train_values = train_values.astype(int)
    train = [[]]
    for k in range(n_train):
        train.append(np.array([px_gauss[k],py_gauss[k],pz_gauss[k],mass_gauss[k],tau_gauss[k]]))
    train.pop(0)
    train = np.asarray(train)
    return train, train_values
