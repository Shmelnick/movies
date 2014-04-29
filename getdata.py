import argparse
import logging
import requests

__author__ = 'nikolayanokhin'


class Movie(object):

    def __init__(self, movie_id, title):
        assert isinstance(title, unicode), "Name {0} is of type {1}".format(title, type(title))
        self.movie_id = movie_id
        self.title = title
        self.actors = set()
        self.directors = set()
        self.synopsis = ""
        self.mpaa_rating = 0
        self.studio = ""
        self.genres = []

    def __mpaa_transform__(self, mpaa_str):
        trans = {"Unrated": 0, "G": 1, "PG": 2, "PG-13": 3, "R": 4, "NC-17": 5}
        return trans[mpaa_str]

    def add_actor(self, actor):
        assert isinstance(actor, unicode), "Tag {0} is of type {1}".format(actor, type(actor))
        self.actors.add(actor)

    def add_director(self, director):
        assert isinstance(director, unicode), "Tag {0} is of type {1}".format(director, type(director))
        self.directors.add(director)

    def __repr__(self):
        return "Movie('{0}', '{1}', actors={2}, genres={3})".format(self.movie_id, self.title.encode('ascii', 'ignore'), self.actors, self.genres)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.movie_id == other.movie_id and self.title == other.title

    def __hash__(self):
        return hash(self.movie_id)


class ApiClient(object):

    API_URL = "http://api.rottentomatoes.com/api/public/v1.0/movies.json"
    MOVIE_URL = "http://api.rottentomatoes.com/api/public/v1.0/movies/{}.json"

    def __init__(self, api_key):
        self.api_key = api_key

    def _load(self, **kwargs):
        params = dict(kwargs)
        params["apikey"] = self.api_key
        response = requests.get(self.API_URL, params=params).json()
        if response and "Error" in response:
            raise ValueError(response.get("Error", "Unknown error"))
        else:
            return response


    def _load_movie(self, movie_id, **kwargs):
        params = dict(kwargs)
        params["apikey"] = self.api_key
        response = requests.get(self.MOVIE_URL.format(str(movie_id)), params=params).json()
        if response and "Error" in response:
            raise ValueError(response.get("Error", "Unknown error"))
        else:
            return response


    def search_movies(self, keyword, page_limit=50):
        logging.debug("Searching movies by keyword '%s'", keyword)
        response = self._load(q=keyword, page_limit=page_limit)
        if response:
            movies = response.get("movies")
            if movies:
                for result in movies:
                    movie_id = result.get("id")
                    title = result.get("title")
                    if movie_id and title:
                        movie = Movie(movie_id, title)
                        movie.synopsis = result.get("synopsis", "")
                        # Actors
                        cast = result.get("abridged_cast", [])
                        for actor in cast:
                            actor_name = actor.get("name")
                            if actor_name:
                                movie.add_actor(actor_name)

                    # Load extra movie information
                    m = self._load_movie(movie_id)
                    directors = result.get("abridged_directors", [])
                    for director in directors:
                        director_name = director.get("name")
                        if (director_name:
                            movie.add_director(director_name)
                    movie.genres = m.get("genres")
                    movie.mpaa_rating = movie.__mpaa_transform__(m.get("mpaa_rating"))
                    # Studio
                    movie.studio = m.get("studio")
                    yield movie


def main():
    # Set up logging
    logging.basicConfig(level=logging.ERROR, format="[%(asctime)-15s] %(message)s")
    print "welcome to the IMDB clustering example"

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--key", action="store", help="Rotten tomatoes account api key", required=True)
    parser.add_argument("keywords", nargs='+', help="The keywords used to search movies")
    args = parser.parse_args()

    for keyword in args.keywords:
        logging.debug("Searching movies for keyword '%s'", keyword)
        client = ApiClient(args.key)
        for movie in client.search_movies(keyword):
            print movie


if __name__ == "__main__":
    main()
