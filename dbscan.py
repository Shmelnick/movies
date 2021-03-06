
import numpy
import numpy.random
import sklearn.datasets
import matplotlib.pylab as pylab
from sklearn.base import BaseEstimator
from getdata import Movie
import csv
import time
import datacomparison
import argparse

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
        self.distances_computed = False
        self.compute_distances(x)
        self.predicted_cluster = self.dbscan(x, self.eps, self.min_samples)
        return self

    def compute_distances(self, x):
        """
        Pre-save matrix of distances between every pair of Movies
        """
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
        """
        Find all neighbours of point with index obj_i
        Returns indexes of neighbours
        """
        res = list()
        for i in xrange(self.samples_amount):
            if self.distance(obj_i, i) <= eps:
                res.append(i)
        return res

    def dbscan(self, x, eps, min_pts):
        """
        DBSCAN main function with added data preprocessing
        Returns: numpy array of cluster labels - if label==0: noise
        """
        self.eps = eps
        self.min_samples = min_pts
        self.last_cluster_index = 0
        self.samples_amount = len(x)
        self.samples = x
        self.visited = [False]*self.samples_amount
        self.predicted_cluster = [0]*self.samples_amount

        # Find start of new cluster
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
        return numpy.array(self.predicted_cluster).clip(0)

    def expand_cluster(self, i, nbr):
        """
        DBSCAN second function
        """
        self.predicted_cluster[i] = self.last_cluster_index
        index = 0
        # check every element of neighborhood
        while index < len(nbr):
            # el = index in samples
            el = nbr[index]
            if not self.visited[el]:
                self.visited[el] = True
                nbr_j = self.neighbours(el, self.eps)
                if len(nbr_j) >= self.min_samples:
                    # add neighborhood to current cluster
                    for c_el in nbr_j:
                        if c_el not in nbr:
                            nbr.append(c_el)
            if self.predicted_cluster[el] <= 0:
                self.predicted_cluster[el] = self.last_cluster_index
            index += 1

    def distance(self, i1, i2):
        """
        Method tries to find distance in pre-saved matrix - if not - this will be computed
        """
        if self.distances_computed:
            return self.distances[i1][i2]
        return d(self.samples[i1], self.samples[i2])


def d(obj1, obj2):
    """
    Calculate distance between obj1 and obj2 based on 6 features
    """
    d1 = datacomparison.compare_arrays(obj1.abridged_cast_names, obj2.abridged_cast_names, 0.1)
    d2 = datacomparison.compare_singles(obj1.studio, obj2.studio)
    d3 = datacomparison.compare_ratings(obj1.mpaa_rating, obj2.mpaa_rating)
    d4 = datacomparison.compare_singles(obj1.first_director, obj2.first_director)
    d5 = datacomparison.compare_arrays(obj1.critics_consensus, obj2.critics_consensus)
    d6 = datacomparison.compare_runtime(obj1.runtime, obj2.runtime)
    return 10*d1 + 10*d2 + 25*d3 + 10*d4 + 15*d5 + 30*d6


def silhouette(movies, labels, dist):
    """
    Used to check clusterization quality
    e_nikolaev
    """
    n = max(set(labels)) + 1
    all = len(labels)
    clusters = dict([(i, []) for i in xrange(n)])
    zero = 0

    # push not-noise-movies into cluster arrays, count noise amount
    for i in range(all):
        c = labels[i]
        if c != 0:
            clusters[c].append(movies[i])
        else:
            zero += 1

    if zero == all:
        return "No clusters detected"

    s = 0
    for i in xrange(all):
        if labels[i] == 0:
            # noise doesn't count
            continue
        # distance between movie and its cluster
        a = 0
        # distance between movie and one of another clusters
        b = 1000
        for c in clusters:
            if c == 0:
                continue
            if c != labels[i]:
                cur_b = 0
                for el in clusters[c]:
                    cur_b += dist(movies[i], el)
                cur_b /= float(len(clusters[c]))
                if cur_b < b:
                    b = cur_b
            else:
                for el in clusters[c]:
                    a += dist(movies[i], el)
                a /= float(len(clusters[c]))
        if max(a, b) != 0:
            s += (b-a)/float(max(a, b))

    return s/float(all - zero)


def jaccard(a, b):
    """
    Jaccard distance between arrays
    """
    sa = set(a)
    sb = set(b)
    return 1 - len(sa.intersection(sb)) / float(len(sa.union(sb)))


#def generate_data(n):
#    """
#    Generate n clusters with random data
#    """
#    print "Generating data set with %s clusters" % n
#    centers = numpy.random.randint(10, size=(n, 2))
#    return sklearn.datasets.make_blobs(n_samples=500, centers=centers, cluster_std=1)


#def plot_data(x, labels):
#    print "Plotting data set"
#    for label in numpy.unique(labels):
#        colors = pylab.cm.jet(numpy.float(label) / numpy.max(labels + 1))
#        pylab.scatter(x[labels == label, 0], x[labels == label, 1], c=colors)


def get_check_array(movies):
    """
    Returns array with features for check clusterization quality (genres)
    """
    return [m.genres for m in movies]


def read_file():
    """
    Returns list of Movie objects - read from file "data.csv"
    """
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
    print '\nReading finished: '+str(len(movies))
    return movies


def print_films_from_cluster(index, x, predicted):
    res = list()
    for i in xrange(len(predicted)):
        if predicted[i] == index:
            res.append(x[i].title)
    return res


def parse_args():
    parser = argparse.ArgumentParser(description='DBSCAN algorithm')
    parser.add_argument('-e', dest='eps', type=float, default=30, help='Neighborhood of point')
    parser.add_argument('-p', dest='min_pts', type=int, default=9, help='Min amount of neighbor points')
    parser.add_argument('-q', action="store_true", default=False, dest='q', help='Show silhouette quality')
    parser.add_argument('-r', action="store_true", default=False, dest='r', help='Show result clusters')
    return parser.parse_args()


def main():
    print "## Clustering with dbscan ##"
    args = parse_args()

    x = read_file()
    #x, cluster_numbers = generate_data(3)

    my_clf = MyDBSCAN()
    my_clf.compute_distances(x)
    print "distances computed"

    predicted = my_clf.dbscan(x, eps=args.eps, min_pts=args.min_pts)
    #print numpy.unique(predicted)

    y = numpy.bincount(predicted)
    ii = numpy.nonzero(y)[0]
    print zip(ii, y[ii])

    if args.q:
        print "Quality: ", silhouette(get_check_array(x), predicted, jaccard)
    if args.r:
        for i in set(predicted):
            if i == 0:
                continue
            print print_films_from_cluster(i, x, predicted)


    #plot_data(x, predicted)
    #pylab.show()


if __name__ == "__main__":
    main()
