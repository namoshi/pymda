import numpy as np
import numpy.linalg
import klin

"""
Fisher's Discriminant Analysis
Usage: Eval, Evec, tmean, xmean = MyFDA.danal(data, label, K)
"""
def cca(X, Y):


#    print '#### Canonical Correlation Analyais ####'
#    print 'X\n', X
#    print 'Y\n', Y


    xsamples, n = X.shape
    ysamples, m = Y.shape
    
#    print 'xsamples=', xsamples, 'n=', n
#    print 'ysamples=', ysamples, 'm=', m
    
    if (xsamples != ysamples):
        print 'The number of samples are inconsistent'
    else:
        nsamples = xsamples
        
        XMean = sum(X) / float(nsamples)
        YMean = sum(Y) / float(nsamples)
        
#        print 'XMean', XMean
#        print 'YMean', YMean
        
        XX = X - XMean
        YY = Y - YMean
        
        SXX = np.dot(XX.T, XX) / float(nsamples-1)
        SYY = np.dot(YY.T, YY) / float(nsamples-1)
        SXY = np.dot(XX.T, YY) / float(nsamples-1)

        SYYinv = np.linalg.inv(SYY)
        SXYYYXY = np.dot(SXY,np.dot(SYYinv, SXY.T))
        Eval, Evec = klin.pencil(SXYYYXY,SXX)

        idx= Eval.argsort()
        Eval = Eval[idx][::-1]
        Evec = Evec[:,idx][:,::-1]
        
        A = Evec
        
#        print 'sqrt(Eval)', np.sqrt(Eval)
#        print 'A\n', A
        
#        U = np.dot(XX, A)
        
#        print 'U\n', U
        
#        print 'inv(Eval)', 1 / np.sqrt(Eval)

        Evalinv = 1.0 / np.sqrt(Eval)        
        B = np.dot(np.dot(SYYinv, np.dot(SXY.T, A)), np.diag(Evalinv))
        
#        print 'B\n', B
        
#        V = np.dot(YY, B)
        
#        print 'V\n', V
        
        return (A, B, XMean, YMean)
        
        
    
#######################
# Compute Score Vectors for features U and V
# X, Y: input features
# A, B: coefficient matrices obtained by MyPCA
# XMean, YMean: mean vectors of the training samples
#######################
def score(X, Y, A, B, XMean, YMean):
    
    XX = X - XMean
    YY = Y - YMean

    U = np.dot(XX, A)
    V = np.dot(YY, B)


    return (U, V)

        


