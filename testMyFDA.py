def main():
    import numpy as np
    import MyFDA
    import pylab as pl
    from sklearn import datasets

    iris = datasets.load_iris()
    data = iris.data
    label = iris.target
#    irisdata = np.loadtxt("iris.dat",comments='#')
#    data = np.asarray(irisdata[:,1:],dtype=float)
#    label = irisdata[:,0] - 1
#    target_names = iris.target_names

    print 'Size of the data = ', data.shape
    print 'label', label
    print 'data type is ', data.dtype

    n, mdim = data.shape

    eigvalue, eigvector, tmean, xmean = MyFDA.danal(data, label, 3)
#    eigvector, eigvalue, mean = MyPCA.pca(data,mdim)

    print 'Eigen Values are ' 
    print eigvalue

    print 'Eigen Vectors are'
    print eigvector

    print 'Total Mean Vector'
    print tmean

    print 'Class Mean Vectors'
    print xmean

#    print 'data'
#    print data

#    score = MyFDA.score(data, eigvector, tmean)
    score = np.dot(data - tmean, eigvector)

    print 'FDA Scores are'
    print 'score dtype is ', score.dtype
    print score

#    ymean = np.dot(xmean - np.dot(np.array([np.ones(len(xmean))]).T,np.array([tmean])), eigvector)
    ymean = np.dot(xmean - tmean, eigvector)
    
    print 'Mean score vector of each class are'
    print ymean


    classes = range(len(iris.target_names))

    pl.scatter(score[:,0], score[:,1], marker='o', c=label)
    pl.scatter(ymean[:,0], ymean[:,1], marker='x', c=classes)
    pl.show()
 
if __name__=="__main__":
    main()
