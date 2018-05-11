import os
from random import randint
import numpy as np

from calc import Platform
from utils import read_with_cache

if __name__ == '__main__':
    # 载入数据集
    ratings_csv = os.path.abspath(os.path.join('.', 'ml-latest-small', 'ratings.csv'))
    movies_csv = os.path.abspath(os.path.join('.', 'ml-latest-small', 'movies.csv'))
    ratings, movies, hashes = read_with_cache(ratings_csv, movies_csv)

    # 划分数据集，模拟数据分布在两个平台的情景
    offset = randint(len(ratings) / 4, len(ratings) / 2)
    jd = Platform('京东', hashes, data=ratings[:offset])
    ali = Platform('淘宝', hashes, data=ratings[offset:])

    # 将一个平台注册到另一个平台以参与联动计算
    ali.append_subscriber(jd)

    # 假设新到达一个用户，要求产生推荐结果
    _, items_count = ratings.shape
    user_vec = np.random.randint(0, 6, (items_count,))

    print('Recommend for %s...' % user_vec)
    recommendations = ali.recommend(user_vec)

    # 打印预测结果
    print(recommendations)
