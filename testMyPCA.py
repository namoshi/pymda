def main():
    import numpy as np
    import MyPCA
    import pylab as pl
    from sklearn import datasets

    # data = np.loadtxt("iris.dat",comments='#')
    iris = datasets.load_iris()
    data = iris.data
    y = iris.target
    target_names = iris.target_names

    print 'Size of the data = ', data.shape

    n = data.shape[0]
    mdim = data.shape[1]

    eigvector, eigvalue, mean = MyPCA.pca(data,mdim)

    print 'Eigen Values are ' 
    print eigvalue

    print 'Eigen Vectors are'
    print eigvector

    score = MyPCA.score(data, eigvector[:,:2], mean)

    print 'PCA Scores are'
    print score

    pl.scatter(score[:,0], score[:,1], marker='o', c=y)
    pl.show()
 
if __name__=="__main__":
    main()
