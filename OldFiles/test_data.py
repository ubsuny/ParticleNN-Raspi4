import tensorflow as tf
import numpy as np
from ExpodentialDistribution import Exp
from GaussianDistribution import Gauss

def test_data(p,Ntest):
    params = 4

    ETest,ETestVals = Exp(p,Ntest)

    GTest,GTestVals = Gauss(p,Ntest)

    Test = np.zeros((2*Ntest,params))
    Test_vals = np.zeros((2*Ntest))

    Test[:Ntest,:] = ETest
    Test[Ntest:,:] = GTest
    Test_vals[:Ntest] = ETestVals
    Test_vals[Ntest:] = GTestVals
    
    return Test,Test_vals



