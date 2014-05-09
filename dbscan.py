
import numpy
import numpy.random
from sklearn.cluster import Ward
import sklearn.datasets
import matplotlib.pylab as pylab
import scipy.spatial.distance as dist
from sklearn.cluster import dbscan
from sklearn.base import BaseEstimator
from getdata import Movie
import csv
import time
import datacomparison

__author__ = 'a_melnikov'


class MyDBSCAN(BaseEstimator):

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.last_cluster_index = 0
        self.samples_amount = 0
        self.samples = []
        self.visited = []
        self.predicted_cluster = []
        self.distances = []
        self.distances_computed = False

    def fit(self, x):
        self.compute_distances(x)
        self.predicted_cluster = self.dbscan(x, self.eps, self.min_samples)
        return self

    def compute_distances(self, x):
        self.distances_computed = False
        self.samples = x
        self.samples_amount = len(x)
        self.distances = numpy.zeros((self.samples_amount, self.samples_amount))
        for i in xrange(self.samples_amount):
            for j in xrange(i, self.samples_amount):
                self.distances[i][j] = self.distance(i, j)
                if i != j:
                    self.distances[j][i] = self.distances[i][j]
        self.distances_computed = True

    def neighbours(self, obj_i, eps):
        res = list()
        for i in xrange(self.samples_amount):
            if self.distance(obj_i, i) <= eps:
                res.append(i)
        return res

    def dbscan(self, x, eps, min_pts):
        self.eps = eps
        self.min_samples = min_pts
        self.last_cluster_index = 0
        self.samples_amount = len(x)
        self.samples = x
        self.visited = [False]*self.samples_amount
        self.predicted_cluster = [0]*self.samples_amount

        for i in xrange(self.samples_amount):
            if self.visited[i]:
                continue
            self.visited[i] = True
            nbr = self.neighbours(i, eps)
            if len(nbr) < self.min_samples:
                self.predicted_cluster[i] = -1   # Noise
            else:
                self.last_cluster_index += 1
                self.expand_cluster(i, nbr)
        return numpy.array(self.predicted_cluster)

    def expand_cluster(self, i, nbr):
        self.predicted_cluster[i] = self.last_cluster_index
        for j in nbr:
            if not self.visited[j]:
                self.visited[j] = True
                nbr_j = self.neighbours(j, self.eps)
                if len(nbr_j) >= self.min_samples:
                    if self.predicted_cluster[j] <= 0:
                        self.predicted_cluster[j] = self.last_cluster_index

    def distance(self, i1, i2):
        if self.distances_computed:
            return self.distances[i1][i2]
        return self.d(self.samples[i1], self.samples[i2])

    def d(self, obj1, obj2):
        d1 = datacomparison.compare_arrays(obj1.abridged_cast_names, obj2.abridged_cast_names)
        d2 = datacomparison.compare_singles(obj1.studio, obj2.studio)
        return (d1**2 + d2**2)**0.5


def generate_data(n):
        print "Generating data set with %s clusters" % n
        centers = numpy.random.randint(10, size=(n, 2))
        return sklearn.datasets.make_blobs(n_samples=500, centers=centers, cluster_std=1)


def plot_data(x, labels):
    print "Plotting data set"
    for label in numpy.unique(labels):
        colors = pylab.cm.jet(numpy.float(label) / numpy.max(labels + 1))
        pylab.scatter(x[labels == label, 0], x[labels == label, 1], c=colors)


def read_file():
    movies = []
    with open('data.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        l = 10
        cur = 0
        data = [row for row in reader]
        n = len(data)/l
        print "Amount of movies: " + str(n)
        time.sleep(2)
        for i in xrange(n):
            ([movie_id], genres, [title], synopsis, [mpaa_rating], [runtime], critics_consensus, abridged_cast_names,
             [first_director], [studio]) = data[cur:cur+l]
            movie = Movie(movie_id, unicode(title))
            movie.genres = genres
            movie.synopsis = synopsis
            movie.mpaa_rating = mpaa_rating
            movie.runtime = runtime
            movie.critics_consensus = critics_consensus
            movie.abridged_cast_names = abridged_cast_names
            movie.first_director = first_director
            movie.studio = studio

            movies.append(movie)

            cur += l
    print '\nDone: '+str(len(movies))
    return movies


def main():
    print "## Clustering with dbscan ##"

    x = read_file()
    #x, cluster_numbers = generate_data(3)

    my_clf = MyDBSCAN()
    my_clf.compute_distances(x)
    print "d comp"
    #print my_clf.distances
    predicted = my_clf.dbscan(x, eps=1, min_pts=50)
    print numpy.unique(predicted)

    y = numpy.bincount(predicted.clip(0))
    ii = numpy.nonzero(y)[0]
    print zip(ii, y[ii])

    #plot_data(x, predicted)
    #pylab.show()

if __name__ == "__main__":
    main()
