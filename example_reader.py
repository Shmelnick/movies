import csv
import time


with open('data.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    l = 10
    cur = 0
    data = [row for row in reader]
    n = len(data)/l
    print n
    time.sleep(2)
    movies = []
    for i in xrange(n):
        ([movie_id], genres, [title], synopsis, [mpaa_rating], [runtime], critics_consensus, abridged_cast_names,
         [first_director], [studio]) = data[cur:cur+l]
        movie = {
            'id':                       movie_id,
            'genres':                   genres,
            'title':	                title,
            'synopsis':	                synopsis,
            'mpaa_rating':              mpaa_rating,
            'runtime':                  runtime,
            'critics_consensus':        critics_consensus,
            'abridged_cast_names':      abridged_cast_names,
            'first_director':           first_director,
            'studio':                   studio
        }
        movies.append(movie)
        print movie['critics_consensus']
        cur += l

    print '\n\ncount: '+str(len(movies))