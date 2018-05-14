import functools
import os
import warnings
from csv import DictReader
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py


def read_from_csv(ratings_csv, movies_csv, tables=4, d=8, size=None):
    """
    读取CSV数据集并生成局部敏感哈希依赖的哈希表
    :param ratings_csv: 评分记录文件位置
    :param movies_csv: 电影名称文件位置
    :param tables: 启用哈希表数量
    :param d: 局部敏感哈希数据降维后的向量长度
    :param size: 返回矩阵尺寸约束
    :return: 用户-电影评分矩阵，电影向量，哈希表组，电影在数据集中对应的原始索引
    """
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
    movie_origin_indexes = np.empty((len(used_movies),))
    for i, movie_id in enumerate(used_movies):
        new_movies.append(movies[movie_id])
        movie_zip_map[movie_id] = i
        movie_origin_indexes[i] = movie_id

    mat = np.zeros((len(ratings), len(used_movies)))
    for user_id, user_movies in ratings.items():
        for movie_id, rating in user_movies.items():
            mat[user_zip_map[user_id], movie_zip_map[movie_id]] = rating

    if size is not None:
        mat = mat[:size[0], :size[1]]
        movie_origin_indexes = movie_origin_indexes[:size[1]]

    _, cols = mat.shape
    hashes = [np.random.uniform(-1, 1, (d, cols)) for _ in range(tables)]

    return mat, new_movies, hashes, movie_origin_indexes


def write_to_hdf5(hdf5_filename, ratings, movie_origin_indexes, hashes, attrs) -> None:
    """
    将从原始数据集读入并生成的数据写入HDF5文件以重复利用
    :param hdf5_filename: HDF5文件名
    :param ratings: 用户-项目评分矩阵
    :param movie_origin_indexes: 电影在数据集中对应的原始索引
    :param hashes: 哈希表组
    :param attrs: 写入HDF5文件的属性
    :return: 无
    """
    with h5py.File(hdf5_filename, 'w') as f:
        f.create_dataset('ratings', data=ratings)
        f.create_dataset('movie_origin_indexes', data=movie_origin_indexes)
        group = f.create_group('hashes')
        for i, hashes_group in enumerate(hashes):
            group.create_dataset('hash_table_%s' % i, data=hashes_group)

        for name, attr in attrs.items():
            f.attrs[name] = attr


def read_movies_by_indexes(movies_csv, movie_origin_indexes):
    """
    根据电影在数据集中对应的原始索引获取电影名称列表
    :param movies_csv: 电影名称文件位置
    :param movie_origin_indexes: 电影在数据集中对应的原始索引
    :return: 电影名称列表
    """
    movies = {}
    with open(movies_csv, 'r', newline='') as f:
        reader = DictReader(f)
        for row in reader:
            movies[int(row['movieId'])] = row['title']

    new_movies = []
    for movie_id in movie_origin_indexes:
        new_movies.append(movies[movie_id])
    return new_movies


def read_with_cache(ratings_csv, movies_csv, tables=4, d=8, hdf5_filename='origin.hdf5', size=None,
                    force_recreate=False):
    """
    尝试从HDF5文件中读取缓存，若失败则重新生成并返回
    :param ratings_csv: 评分记录文件位置
    :param movies_csv: 电影名称文件位置
    :param tables: 启用哈希表数量
    :param d: 局部敏感哈希数据降维后的向量长度
    :param hdf5_filename: HDF5文件名
    :param size: 返回矩阵尺寸约束
    :param force_recreate: 强制重新生成
    :return: 用户-电影评分矩阵，电影向量，哈希表组，电影在数据集中对应的原始索引
    """
    if not force_recreate and os.path.exists(hdf5_filename):
        with h5py.File(hdf5_filename, 'r') as f:
            if ratings_csv == f.attrs['ratings_csv'] and movies_csv == f.attrs['movies_csv']:
                if size is None or (f.attrs['users_size'] == size[0] and f.attrs['movies_size'] == size[1]):
                    movie_origin_indexes = np.array(f['movie_origin_indexes'])
                    return np.array(f['ratings']), read_movies_by_indexes(movies_csv, movie_origin_indexes), \
                           [np.array(hashes_group) for _, hashes_group in f['hashes'].items()], movie_origin_indexes

    ratings, movies, hashes, movie_origin_indexes = read_from_csv(ratings_csv, movies_csv, tables, d, size)

    write_to_hdf5(hdf5_filename, ratings, movie_origin_indexes, hashes, {
        'ratings_csv': ratings_csv,
        'movies_csv': movies_csv,
        'users_size': size[0],
        'movies_size': size[1]
    })
    return ratings, movies, hashes, movie_origin_indexes


def evaluate(x, y):
    """
    计算向量MAE值
    :param x: 向量1
    :param y: 向量2
    :return: 向量1和向量2的MAE值
    """
    if x is not None and y is not None:
        assert len(x) == len(y) != 0
        x, y = np.asarray(x), np.asarray(y)
        index = np.logical_and(x > 0, y > 0)
        x, y = x[index], y[index]
        sum_of_diff = np.add.reduce(np.abs(x - y))
        return sum_of_diff / len(x)
    else:
        return 0


def mean_1d(x, remove=0):
    """
    去除最值后取均值
    :param x: 待求向量
    :param remove: 去除极值数量
    :return: 向量元素均值
    """
    assert remove > -1
    if remove == 0:
        return x.mean(axis=0)
    else:
        x_max = x.max()
        x_min = x.min()
        index = np.logical_or(x == x_max, x == x_min) == False
        return mean_1d(x[index])


def plot_and_save(x, ys, legends, x_label, y_label, figure_name) -> None:
    """
    绘制并存储折线图
    :param x: X座标值串
    :param ys: Y座标值串组
    :param legends: 图例
    :param x_label: X座标标签
    :param y_label: Y座标标签
    :param figure_name: 图标题
    :return: 无
    """
    assert len(ys) == len(legends) > 0
    plt.figure()
    for i, y in enumerate(ys):
        plt.plot(x, y, label=legends[i])
    plt.legend(legends)
    plt.title(figure_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig('%s.png' % os.path.join('assets', figure_name))


def log_duration(func):
    """
    打印函数运行时间装饰器
    :param func: 被修饰函数
    :return: 修饰后函数
    """
    @functools.wraps(func)
    def timed(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        print('%s Start at %s, End at %s. Duration: %s.' % (func.__name__, start, end, end - start))
        return result
    return timed
