"""
K-means algorythm implementation with modifications

"""

__author__ = "Vladislav Kulikov"

import numpy as np
from sklearn.base import BaseEstimator

class KMeans:
    "Clusterizator by K-means algorythm"

    def __init__( self, n_clusters = 8 ):
        self.n_clusters = n_clusters
        self.mass_centers = []
            

    def __norm2(self, obj):
        return np.sum(obj ** 2)


    def fit_predict( self, X, eps = 0.001 ):
        "KMeans method: fit_predict(X) -> [n_samples]"

        k = self.n_clusters
        size = X.shape[0]
        # Random initialize start mass centers
        mc = [X[i] for i in np.random.randint(0, size, k)]
        self.mass_centers = np.zeros((k, X.shape[1]))
        rnk = np.zeros(size)

        ### Other variant: while False in (rnkold == rnknew):
        while self.__norm2(self.mass_centers - mc) > eps:
            t = self.mass_centers
            self.mass_centers = mc
            mc = t
            # Expectation step
            for i in xrange(size):
                rnk[i] = np.argmin([self.__norm2(X[i] - mu) \
                                    for mu in self.mass_centers])
            # Maximization step
            for i in xrange(k):
                kx = X[rnk == i]
                mc[i] = np.sum(kx, 0) / float(len(kx))
                
        return rnk


    def fit( self, X ):
        "KMeans method: fit(X) -> self"

        self.fit_predict(X)
        return self


    def predict( self, X ):
        "KMeans method: predict(X) -> [n_samples]"

        size = X.shape[0]
        rnk = np.zeros(size)
        for i in xrange(size):
            rnk[i] = np.argmin([self.__norm2(X[i] - mu) \
                                   for mu in self.mass_centers])
        return rnk
        

    def score( self ):
        pass

        
class KMeansMiniBatch:
    "Clusterizator by K-means mini-batch algorythm"

    def __init__( self, n_clusters = 8 ):
        self.n_clusters = n_clusters
        self.mass_centers = []
            

    def __norm2(self, obj):
        return np.sum(obj ** 2)


    def fit_predict( self, X, b = None ):
        "KMeans method: fit_predict(X) -> [n_samples]"

        k = self.n_clusters
        size = X.shape[0]
        if b == None:
            b = size / (3 * k)
        # Random initialize start mass centers
        mc = [X[i] for i in np.random.randint(0, size, k)]
        self.mass_centers = np.zeros((k, X.shape[1]))
        rnk = np.zeros(size)

        ### Other variant: while False in (rnkold == rnknew):
        while self.__norm2(self.mass_centers - mc) > eps:
            t = self.mass_centers
            self.mass_centers = mc
            mc = t
            workX = [X[i] for i in np.random.randint(0, size, b)]
            # Expectation step
            for i in xrange(size):
                rnk[i] = np.argmin([self.__norm2(workX[i] - mu) \
                                    for mu in self.mass_centers])
            # Maximization step
            for i in xrange(k):
                kx = workX[rnk == i]
                mc[i] = np.sum(kx, 0) / float(len(kx))
                
        return rnk



    def fit( self, X ):
        "KMeans method: fit(X) -> self"

        self.fit_predict(X)
        return self


    def predict( self, X ):
        "KMeans method: predict(X) -> [n_samples]"

        size = X.shape[0]
        rnk = np.empty(size)
        for i in xrange(size):
            rnk[i] = np.argmin([self.__norm2(X[i] - mu) \
                                   for mu in self.mass_centers])
        return rnk
        
    def score( self ):
        pass

        
