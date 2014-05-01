import numpy as np
import numpy.linalg
import klin

"""
Fisher's Discriminant Analysis
Usage: Eval, Evec, tmean, xmean = MyFDA.danal(data, label, K)
"""
def su3(data):

    m, n = data.shape

    if (m < n):
        minmn = m
    else:
        minmn = n
    
    total = np.sum(A)

#    R = np.zeros((m,n))

    R = A / float(total)
    
####################

	P = np.dot(R, np.ones((n)))
    
    Q = np.dot(np.ones((m)), R)


    return (R, P, Q)


###
# Compute Scores Vectors of FDA for the given features
# features: input features
# evec: eigen vectors of FDA
# total mean: vector of the training samples
###
def score(features, evec, tmean):

#    diff = features - np.dot(np.array([np.ones(len(features))]).T,np.array([tmean]))
    diff = features - tmean
    return np.dot(diff, evec)



