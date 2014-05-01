import sys
import numpy as np
import numpy.linalg

"""
Kurita's Linear Algebra Routines
"""

###
# Solve the Eigen Value Problem (desending order)
# A Evec = Evec Eval
# Usage:  Eval, Evec = eigen(A)
###
def eigen(A):

    shapeA = A.shape

    if not (shapeA[0] == shapeA[1]):
        print>>sys.stderr, 'Error (eigen): the matrix A must be square.'
    else:

        Eval, Evec = np.linalg.eigh(A)
        idx = Eval.argsort()
        Eval = Eval[idx][::-1]
        Evec = Evec[:,idx][:,::-1]

    return (Eval, Evec)


###
# Generalized Eigen Problem
# A EV = B EV L
# Usage: Eval, Evec = pencil(A,B)
###
def pencil(A,B):

    EPS = 1.0e-10

    shapeA = A.shape
    shapeB = B.shape

#    print 'shapeA=', shapeA
#    print 'shapeB=', shapeB

    if shapeA[0] == shapeA[1] and shapeB[0] == shapeB[1] and shapeA[0] == shapeB[0]:
        m = shapeA[0]
        EV,U = eigen(B)
        
        r1 = EV[0]
        if (r1 > EPS):
            for i in range(m):
                re = np.absolute(EV[i])
                if (re >= EPS * r1):
                    EV[i] = 1.0 / np.sqrt(re)
                else:
                    EV[i] = 0.0
        
        U = np.multiply(U,EV)

#        print 'U'
#        print U
        
        R = np.dot(np.dot(U.T, A), U)

        Eval, S = eigen(R)

        Evec = np.dot(U,S)

    else:
        print>>std.error, 'Error (pencil): The matrix must be the same size'
        

    return (Eval, Evec)


###
# Compute RBF Kernel (Gram) Matrix
# 
# Usage: gram = klin.KernelMatrix(X_train, gamma = 1.0)
#        gram_test = klin.KernelMatrix(X_train, X_test, gamma = 0.1) 
###
def KernelMatrix(X_train, X_test = [], gamma = 1.0):
    import numpy as np
    import scipy as scip
    from scipy.spatial.distance import pdist, squareform
    
    if X_test == []:
        pairwise_dists = squareform(pdist(X_train, 'euclidean'))
        gram = scip.exp(-gamma*(pairwise_dists ** 2))
    else:
        if X_train.shape[1] == X_test.shape[1]:
            gram = np.zeros((len(X_test),len(X_train)))
            for j in range(len(X_train)):
                dif = X_test - np.dot(np.ones((len(X_test),1)), [X_train[j,:]])
                gram[:,j] = scip.exp(-gamma* np.dot((dif ** 2),np.ones(((X_test.shape[1]),1))))[:,0]
        else:
            print 'Error: Feature dimensions are not consistent!!\n'
            gram = []


    return gram


