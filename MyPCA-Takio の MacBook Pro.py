import sys
import numpy as np
import numpy.linalg

"""Principal Component Analysis
   eigenvectors:row vectors
   eigenvalues:sorted eigen values
   mean:mean vector (row vector)
"""
def pca(src, dim=0, alpha=0.0):

    n, m = src.shape

    if (dim == 0):
        dim = m
 
    # Transpose after subtracting the mean vector from each sample
    mu = sum(np.asarray(src,dtype=np.float)) / float(n)
    src_m = (src - mu).T
 
    if n < m: # number of sample < dimention

        if n < dim: dim=n
        
        print>>sys.stderr,"covariance matrix ...",
        cov=np.dot(src_m.T, src_m)/float(n) + alpha * np.diag(np.ones((n)))
        print>>sys.stderr,"done"
 
        print>>sys.stderr,"eigen value decomposition ...",
        Eval, Evec = np.linalg.eig(cov)
        idx= Eval.argsort()
        Eval = Eval[idx][::-1]
        Evec = Evec[:,idx][:,::-1]
        print>>sys.stderr,"done"
 
        vm=np.dot(src_m, Evec)
        for i in range(len(Eval)):
            if Eval[i]<=0:
                Evec=Evec[:,:i]
                Eval=Eval[:i]
                if dim < i: dim=i
                break

        vm[:,i]=vm[:,i]/np.sqrt(n*Eval[i])
 
    else: # number of sample >= dimention
 
        if m < dim: dim=m
        cov=np.dot(src_m, src_m.T)/float(n) + alpha * np.diag(np.ones((m)))
        Eval, vm = np.linalg.eig(cov)
        idx= Eval.argsort()
        Eval = Eval[idx][::-1]
        vm = vm[:,idx][:,::-1]
 
    return (vm[:,:dim], Eval[:dim], mu)
 
#######################
# Compute Score Vectors for features
# features: input features
# evectors: eigenvectors obtained by MyPCA
# mean: mean vector of the training samples
#######################
def score(features, evectors, mean):
    diff = features - np.dot(np.array([np.ones(len(features))]).T,np.array([mean]))
#    print features
    return np.dot(diff,evectors)
