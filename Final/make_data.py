'''make_data.py generates the data to be used for the training
and testing data sets, using expodential_distribution.py and
gaussian_distribution.py.  Also shuffles the data.'''
import numpy as np
import tensorflow as tf
from expodential_distribution import exp
from gaussian_distribution import gauss

def make_data(momentum, n_train):
    '''Particle Neural Network: Generates simulation Particle Data,
    to be used to either test or train the Particle Neural Networks.'''
    params = 5
    exp_train,exp_train_vals = exp(momentum,n_train)
    gauss_train,gauss_train_vals = gauss(momentum,n_train)
    train = np.zeros((2*n_train,params+1))
    train[:n_train,0:params] = exp_train
    train[n_train:,0:params] = gauss_train
    train[:n_train,params] = exp_train_vals
    train[n_train:,params] = gauss_train_vals
    buffer_size = n_train*2
    batch_size = n_train*2
    train_dataset = tf.data.Dataset.from_tensor_slices(train).shuffle(buffer_size).batch(batch_size)
    for i in train_dataset.take(-1):
        train = i[0:,0:params]
        train_vals = i[0:,params]
    train = np.array(train)
    train_vals = np.array(train_vals)
    return train, train_vals
