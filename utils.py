from csv import DictReader
import os
import numpy as np
import h5py


def read_from_csv(ratings_csv, movies_csv, tables=4, d=8):
    ratings = {}
    with open(ratings_csv, 'r', newline='') as f:
        reader = DictReader(f)
        for row in reader:
            ratings.setdefault(int(row['userId']), {})
            ratings[int(row['userId'])][int(row['movieId'])] = float(row['rating'])

    movies = {}
    with open(movies_csv, 'r', newline='') as f:
        reader = DictReader(f)
        for row in reader:
            movies[int(row['movieId'])] = row['title']

    used_movies = set()
    for _, user_movies in ratings.items():
        for movie_id in user_movies:
            used_movies.add(movie_id)

    user_zip_map = {}
    for i, user_id in enumerate(ratings):
        user_zip_map[user_id] = i

    movie_zip_map = {}
    new_movies = []
    for i, movie_id in enumerate(used_movies):
        new_movies.append(movies[movie_id])
        movie_zip_map[movie_id] = i

    mat = np.zeros((len(ratings), len(used_movies)))
    for user_id, user_movies in ratings.items():
        for movie_id, rating in user_movies.items():
            mat[user_zip_map[user_id], movie_zip_map[movie_id]] = rating

    _, cols = mat.shape
    hashes = [np.random.uniform(-1, 1, (d, cols)) for _ in range(tables)]

    return mat, new_movies, hashes


def write_to_hdf5(hdf5_filename, ratings, movies, hashes, attrs) -> None:
    with h5py.File(hdf5_filename, 'w') as f:
        f.create_dataset('ratings', data=ratings)
        f.create_dataset('movies', data=movies)
        group = f.create_group('hashes')
        for i, hashes_group in enumerate(hashes):
            group.create_dataset('hash_table_%s' % i, data=hashes_group)

        for name, attr in attrs.items():
            f.attrs[name] = attr


def read_with_cache(ratings_csv, movies_csv, tables=4, d=8, hdf5_filename='data.hdf5', force_recreate=False):
    if not force_recreate and os.path.exists(hdf5_filename):
        with h5py.File(hdf5_filename, 'r') as f:
            if ratings_csv == f.attrs['ratings_csv'] and movies_csv == f.attrs['movies_csv']:
                return np.array(f['ratings']), np.array(f['movies']),\
                       [np.array(hashes_group) for hashes_group in f['hashes']]

    ratings, movies, hashes = read_from_csv(ratings_csv, movies_csv, tables, d)

    write_to_hdf5(hdf5_filename, ratings, movies, hashes, {
        'ratings_csv': ratings_csv,
        'movies_csv': movies_csv
    })
    return ratings, movies, hashes
