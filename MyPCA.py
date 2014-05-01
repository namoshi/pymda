import sys
import numpy as np
import numpy.linalg

"""Principal Component Analysis
   eigenvectors:row vectors
   eigenvalues:sorted eigen values
   mean:mean vector (row vector)
"""
def pca(src,dim):

    shape=src.shape
 
    # Transpose after subtracting the mean vector from each sample
    mu = (sum(np.asarray(src,dtype=np.float)) / float(len(src)))
    src_m = (src - mu).T
 
    if shape[0] < shape[1]: # number of sample < dimention

        if shape[0] < dim: dim=shape[0]
        
        print>>sys.stderr,"covariance matrix ...",
        n=np.dot(src_m.T, src_m)/float(shape[0])
        print>>sys.stderr,"done"
 
        print>>sys.stderr,"eigen value decomposition ...",
        l,v=np.linalg.eig(n)
        idx= l.argsort()
        l = l[idx][::-1]
        v = v[:,idx][:,::-1]
        print>>sys.stderr,"done"
 
        vm=np.dot(src_m, v)
        for i in range(len(l)):
            if l[i]<=0:
                v=v[:,:i]
                l=l[:i]
                if dim < i: dim=i
                break

        vm[:,i]=vm[:,i]/np.sqrt(shape[0]*l[i])
 
    else: # number of sample >= dimention
 
        if shape[1] < dim: dim=shape[1]
        cov=np.dot(src_m, src_m.T)/float(shape[0])
        l,vm = np.linalg.eig(cov)
        idx= l.argsort()
        l = l[idx][::-1]
        vm = vm[:,idx][:,::-1]
 
    return (vm[:,:dim], l[:dim], mu)
 
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
