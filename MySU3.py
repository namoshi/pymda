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
    
    XMEAN = np.zeros((K, nc))
#
#    for i in range(nr):
#        k = label[i];
#        NS[k] = NS[k] + 1
#        for j in range(nc):
#            XMEAN[k,j] = XMEAN[k,j] + data[i,j]
#
#    for k in range(K):
#        omega = NS[k]
#        for j in range(nc):
#            XMEAN[k,j] = XMEAN[k,j] / omega
#
#    print 'Class Means'
#    print XMEAN
    
    NE = np.zeros((K))
    for k in range(K):
        NE[k] = sum(label == k)

    for k in range(K):
        XMEAN[k,:] = np.dot(data[ label == k, :].T, np.ones((int(NE[k])))).T / float(NE[k])

#    print 'Class Means'
#    print XMEAN

    # Between Class Covariance Matrix
#    Sb = np.zeros((nc, nc))
#    for k in range(K):
#        omega = sum(label == k) / float(nr)
#        for i in range(nc):
#            for j in range(nc):
#                Sb[i,j] = Sb[i,j] + omega * (XMEAN[k,i] - TMEAN[i])  * (XMEAN[k,j] - TMEAN[j])
#
#    print 'Between Class Covariance Matrix'
#    print Sb

#    alpha = 0.001

    Sb = np.zeros((nc, nc))

    for k in range(K):
        omega = float(NE[k]) / float(nr)
        xmean_m = XMEAN[k,:] - TMEAN
        Sb = Sb + omega * np.outer(xmean_m, xmean_m)
    
#    print 'Between Class Covariance Matrix'
#    print Sb.dtype
#    print Sb

    Sw = St - Sb

#    Sw = Sw + alpha * np.eye(nc)

#    print 'Within Class Covariance Matrix'
#    print Sw.dtype
#    print Sw

    Eval, Evec = klin.pencil(Sb,Sw)


    return (Eval[:ldim], Evec[:,:ldim], TMEAN, XMEAN)


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



