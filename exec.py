import os
from datetime import datetime
from random import randint

import numpy as np
from matplotlib import pyplot as plt

from calc import Platform
from utils import read_with_cache, evaluate

__author__ = 'i@AngelMsger.Com'


def test_lsh(collection, lsh_hashes, offset=64, step=64, test_count=64) -> None:
    """
        测试基于局部敏感哈希的协同过滤推荐并分析实时性差异
        :param collection: 评分矩阵
        :param lsh_hashes: 局部敏感哈希随机矩阵
        :param offset: 起始位置
        :param step: 步长
        :param test_count: 测试循环次数
        :return: 无
    """
    users_size, items_size = collection.shape

    horizontal = np.arange(offset, users_size, step)
    vertical_using_lsh = np.empty((test_count, len(horizontal)))
    vertical_not_using_lsh = np.empty((test_count, len(horizontal)))
    errors_using_lsh = np.empty((test_count, len(horizontal)))
    errors_not_using_lsh = np.empty((test_count, len(horizontal)))

    for i in range(test_count):
        print('%s...' % i)
        user_vec = np.random.randint(0, 6, (items_size,))
        for j, value in enumerate(horizontal):
            amazon = Platform('亚马逊', lsh_hashes, data=collection[:value])
            start = datetime.now()
            result = amazon.recommend(user_vec)
            end = datetime.now()
            errors_using_lsh[i, j] = evaluate(user_vec, result)
            vertical_using_lsh[i, j] = (end - start).microseconds

            start = datetime.now()
            result = amazon.recommend(user_vec, use_lsh=False)
            end = datetime.now()
            errors_not_using_lsh[i, j] = evaluate(user_vec, result)
            vertical_not_using_lsh[i, j] = (end - start).microseconds

    plt.plot(horizontal, vertical_using_lsh.mean(axis=0), label='Using LSH')
    plt.plot(horizontal, vertical_not_using_lsh.mean(axis=0), label='NOT Using LSH')
    plt.legend(['Using LSH', 'NOT Using SLH'])
    plt.xlabel('Size of DataSet')
    plt.ylabel('Time Cost (ms)')
    plt.imsave(os.path.join('assets', 'effectiveness.png'))

    plt.plot(horizontal, errors_using_lsh.mean(axis=0), label='Using LSH')
    plt.plot(horizontal, errors_not_using_lsh.mean(axis=0), label='NOT Using LSH')
    plt.legend(['Using LSH', 'NOT Using SLH'])
    plt.xlabel('Size of DataSet')
    plt.ylabel('MAE')
    plt.imsave(os.path.join('assets', 'mae.png'))


def test_he(collection, lsh_hashes) -> None:
    """
    测试基于同态加密的多方参与的分布式协同过滤推荐
    :param collection: 评分矩阵
    :param lsh_hashes: 局部敏感哈希随机矩阵
    :return: 无
    """
    # 划分数据集，模拟数据分布在两个平台的情景
    offset = randint(len(collection) // 4, len(collection) // 2)
    jd = Platform('京东', lsh_hashes, data=collection[:offset])
    ali = Platform('淘宝', lsh_hashes, data=collection[offset:])

    # 将一个平台注册到另一个平台以参与联动计算
    ali.append_subscriber(jd)

    # 假设新到达一个用户，要求产生推荐结果
    _, items_size = collection.shape
    user_vec = np.random.randint(0, 6, (items_size,))

    print('Recommend for %s...' % user_vec)
    recommendations = ali.recommend(user_vec, apply_for_subscribers=True)

    # 打印预测结果
    print(recommendations)


if __name__ == '__main__':
    # 载入数据集
    ratings_csv = os.path.abspath(os.path.join('.', 'ml-latest-small', 'ratings.csv'))
    movies_csv = os.path.abspath(os.path.join('.', 'ml-latest-small', 'movies.csv'))
    ratings, _, hashes, _ = read_with_cache(ratings_csv, movies_csv)

    test_lsh(ratings, hashes)
