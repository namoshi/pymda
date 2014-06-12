def main():
    import numpy as np
    import MyCCA
    import pylab as pl

    student = np.loadtxt("student.csv", delimiter=',')

    print 'Size of the data = ', student.shape

    X = student[:,0:2]
    Y = student[:,2:4]

    print 'X\n', X
    print 'Y\n', Y
    
    print
    print
    print 'Canonical Correlation Analysis'
    
    A, B, XMean, YMean = MyCCA.cca(X, Y)
    
    U, V = MyCCA.score(X, Y, A, B, XMean, YMean)
    
    print 'A\n', A
    print 'B\n', B
    print 'U\n', U
    print 'V\n', V

    pl.scatter(U[:,0], V[:,0], marker='o', c='red')
    pl.title('CCA (1st component)')
    pl.xlabel('U1')
    pl.ylabel('V1')
    pl.show()

    pl.scatter(U[:,1], V[:,1], marker='x', c='blue')
    pl.title('CCA (2nd component)')
    pl.xlabel('U2')
    pl.ylabel('V2')
    pl.show()
 
if __name__=="__main__":
    main()