import argparse
import logging
import requests
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import PorterStemmer
import time
import string

KEY = "efsq4zyub5xzs2sa57hg4b6d"


class Movie(object):

    def __init__(self, movie_id, title):

        assert isinstance(title, unicode), "Name {0} is of type {1}".format(title, type(title))
        self.movie_id = movie_id
        self.title = title
        self.synopsis = ""
        self.mpaa_rating = ""
        self.runtime = -1
        self.genres = []
        self.critics_consensus = ""
        self.abridged_cast_names = []
        self.first_director = ""
        self.studio = ""

    def __repr__(self):
        return "Movie('{0}', 'title={1}', genres={2}, mpaa_rating={3}, runtime={4}, " \
                "critics_consensus={5}, abridged_cast_names={6}, first_director={7}, " \
                "studio={8})".format(self.movie_id, self.title.encode('ascii',
                'ignore'), self.genres, self.mpaa_rating, self.runtime,
                self.critics_consensus, self.abridged_cast_names, self.first_director, self.studio)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.movie_id == other.movie_id and self.title == other.title

    def __hash__(self):
        return hash(self.movie_id)

    def to_csv(self):
        return [[self.movie_id], self.genres, [self.title], self.synopsis, [self.mpaa_rating], [self.runtime],
                self.critics_consensus, self.abridged_cast_names, [self.first_director], [self.studio]]


class ApiClient(object):

    API_URL = "http://api.rottentomatoes.com/api/public/v1.0/movies.json"
    MOVIE_URL = "http://api.rottentomatoes.com/api/public/v1.0/movies/{}.json"

    def __init__(self):
        self.api_key = KEY
        self.tokenizer = nltk.WordPunctTokenizer()
        self.stm = PorterStemmer()

    def _load(self, **kwargs):
        """
        Loads list of movies via filter
        """
        params = dict(kwargs)
        params["apikey"] = self.api_key
        response = requests.get(self.API_URL, params=params).json()
        if response and "Error" in response:
            raise ValueError(response.get("Error", "Unknown error"))
        else:
            return response

    def _load_movie(self, movie_id, **kwargs):
        """
        Loads extra movie information such as directors, genres, etc.
        """
        params = dict(kwargs)
        params["apikey"] = self.api_key
        response = requests.get(self.MOVIE_URL.format(str(movie_id)), params=params).json()
        if response and "Error" in response:
            raise ValueError(response.get("Error", "Unknown error"))
        else:
            return response

    def normalize(self, text):
        tokens = list()
        for token in self.tokenizer.tokenize(text.lower()):

            # Excludes stopwords, punctuation; stemming
            if token in stopwords.words('english'):
                continue
            token = self.stm.stem(token)
            if token.isalpha():
                tokens.append(token)

        return tokens

    def get_extra_params(self, movie_id, movie):
        """
        Saves extra features of movie
        """
        m = self._load_movie(movie_id)
        if (m.has_key('genres') and
                m.has_key('runtime') and
                m.has_key('critics_consensus') and
                m.has_key('abridged_cast') and
                m.has_key('abridged_directors') and
                m.has_key('studio')):
            movie.genres = m.get("genres")
            movie.runtime = m.get("runtime")
            movie.critics_consensus = self.normalize(m.get("critics_consensus"))
            movie.abridged_cast_names = [ac['name'] for ac in m.get("abridged_cast")]
            try:
                movie.first_director = m.get("abridged_directors")[0]['name']
            # This never happened: check type of exception
            except ValueError:
                return False
            movie.studio = m.get("studio")                        
            return True
        return False

    def search_movies(self, keyword, movie_ids, page_limit=50):
        #DBG
        logging.debug("Searching movies by keyword '%s'", keyword)

        # Get list of movies
        response = self._load(q=keyword, page_limit=1, page=1)
        n = response.get("total")

        # Load all 25 pages x 50 movies
        for i in xrange(min(n/page_limit, 25)):
            response = self._load(q=keyword, page_limit=page_limit, page=i+1)
            if response:
                movies = response.get("movies")
                if movies:
                    for result in movies:
                        movie_id = result.get("id")
                        print movie_id

                        if not movie_id or movie_id in movie_ids:
                            continue
                        movie_ids.add(movie_id)

                        title = result.get("title")
                        synopsis = result.get("synopsis")
                        # Convert rating into linear scale [0-4]
                        rating = self.set_rating(result.get("mpaa_rating"))

                        if title and rating >= 0:
                            movie = Movie(movie_id, title)
                            if not synopsis:
                                movie.synopsis = ['EMPTY']
                            else:
                                movie.synopsis = self.normalize(synopsis)
                            movie.mpaa_rating = rating

                            # Load extra movie information
                            if self.get_extra_params(movie_id, movie):
                                yield movie

    @staticmethod
    def set_rating(rating):
        if rating == 'G':
            return 0
        elif rating == 'PG':
            return 1
        elif rating == 'PG-13':
            return 2
        elif rating == 'R':
            return 3
        elif rating == 'NC-17':
            return 4
        else:
            return -1


def main():
    # Set up logging
    logging.basicConfig(level=logging.ERROR, format="[%(asctime)-15s] %(message)s")

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("keywords", nargs='+', help="The keywords used to search movies")
    args = parser.parse_args()

    csvfile = open('data.csv', 'wb')
    writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    client = ApiClient()
    movie_ids = set()
    for keyword in args.keywords:
        print "Start searching for " + keyword
        for movie in client.search_movies(keyword, movie_ids):
            writer.writerows(movie.to_csv())


if __name__ == "__main__":
    main()