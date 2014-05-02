import numpy as np
import numpy.linalg
import klin

"""
Fisher's Discriminant Analysis
Usage: Eval, Evec, tmean, xmean = MyFDA.danal(data, label, K)
"""
def su3(data):
    
    EPS = 1.0e-6

    m, n = data.shape

    if (m < n):
        minmn = m
    else:
        minmn = n
    
    total = np.sum(data)

#    R = np.zeros((m,n))

    R = data / float(total)
    
####################

    P = np.dot(R, np.ones((n)))
    Pinv = np.zeros((m))
    for i in range(m):
        if (P[i] >= EPS):
            Pinv[i] = 1.0 / P[i]

    Q = np.dot(np.ones((m)), R)
    Qinv = np.zeros((n))
    for i in range(n):
        if (Q[i] >= EPS):
            Qinv[i] = 1.0 / Q[i]
    
#    print 'R\n', R
#    print 'P\n', P
#    print 'Q\n', Q
    
    # Solve the eigen problem with smaller dimension
    if (m <= n):
        # R Qinv R^T        
        UU = np.dot(np.dot(R, np.diag(Qinv)), R.T)
        
        W = np.diag(P)
        
        Eval, Evec = klin.pencil(UU, W)
        
        E = Eval[1:]
        U = Evec[:,1:]
        Einv = np.zeros(m-1)
        for i in range(m-1):
            if (E[i] >= EPS):
                Einv[i] = 1.0 / np.sqrt(E[i])

#        print 'R\n', R
#        print 'Einv\n', Einv
        
        V = np.dot(np.dot(np.dot(np.diag(Qinv), R.T), U), np.diag(Einv))
        
    else:
        # VV = R^T Pinv R
        VV = np.dot(np.dot(R.T, np.diag(Pinv)), R)
        
        W = np.diag(Q)
        
        Eval, Evec = klin.pencil(VV, W)
    
        E = Eval[1:]
        V = Evec[:,1:]
        Einv = np.zeros(n-1)

        for i in range(n-1):
            if (E[i] >= EPS):
                Einv[i] = 1.0 / np.sqrt(E[i])
          
        U = np.dot(np.dot(np.dot(np.diag(Pinv), R), V), np.diag(Einv))
    
    # remove the first eigen value and eigen vector
    return (E, U, V)


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



