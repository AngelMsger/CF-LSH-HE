import os
from datetime import datetime

import numpy as np

from calc import Platform
from utils import read_with_cache, evaluate, plot_and_save

__author__ = 'i@AngelMsger.Com'


def test_lsh(collection, lsh_hashes, offset=32, step=32, test_count=8) -> None:
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

    size_of_data_set = np.arange(offset, users_size, step)
    effectiveness_using_lsh = np.empty((test_count, len(size_of_data_set)))
    effectiveness_not_using_lsh = np.empty((test_count, len(size_of_data_set)))
    mae_using_lsh = np.empty((test_count, len(size_of_data_set)))
    mae_not_using_lsh = np.empty((test_count, len(size_of_data_set)))

    for i in range(test_count):
        print('%s...' % i)
        user_vec = np.random.randint(0, 6, (items_size,))
        for j, value in enumerate(size_of_data_set):
            amazon = Platform('亚马逊', lsh_hashes, data=collection[:value])
            start = datetime.now()
            result = amazon.recommend(user_vec)
            end = datetime.now()
            mae_using_lsh[i, j] = evaluate(user_vec, result)
            effectiveness_using_lsh[i, j] = (end - start).microseconds

            start = datetime.now()
            result = amazon.recommend(user_vec, use_lsh=False)
            end = datetime.now()
            mae_not_using_lsh[i, j] = evaluate(user_vec, result)
            effectiveness_not_using_lsh[i, j] = (end - start).microseconds

    legends = ['Using LSH', 'NOT Using LSH']
    effectiveness_mean_using_lsh = effectiveness_using_lsh.mean(axis=0)
    effectiveness_mean_not_using_lsh = effectiveness_not_using_lsh.mean(axis=0)
    plot_and_save(size_of_data_set, [effectiveness_mean_using_lsh, effectiveness_mean_not_using_lsh], legends,
                  'Size of DataSet', 'Time Cost (ms)', 'Effectiveness_LSH')
    mae_mean_using_lsh = mae_using_lsh.mean(axis=0)
    mae_mean_not_using_lsh = mae_not_using_lsh.mean(axis=0)
    plot_and_save(size_of_data_set, [mae_mean_using_lsh, mae_mean_not_using_lsh], legends,
                  'Size of DataSet', 'MAE', 'MAE_LSH')


def test_he(collection, lsh_hashes, test_count=64) -> None:
    """
    测试基于同态加密的多方参与的分布式协同过滤推荐
    :param collection: 评分矩阵
    :param lsh_hashes: 局部敏感哈希随机矩阵
    :param test_count: 测试循环次数
    :return: 无
    """
    # 划分数据集，模拟数据分布在两个平台的情景
    offset = len(collection) // 2
    jd = Platform('京东', lsh_hashes, data=collection[:offset])
    ali = Platform('淘宝', lsh_hashes, data=collection[offset:])

    # 将一个平台注册到另一个平台以参与联动计算
    ali.append_subscriber(jd)

    _, items_size = collection.shape

    effectiveness_using_he = np.empty((test_count,))
    effectiveness_not_using_he = np.empty((test_count,))
    mae_using_he = np.empty((test_count,))
    mae_not_using_he = np.empty((test_count,))

    for i in range(test_count):
        # 假设新到达一个用户，要求产生推荐结果
        user_vec = np.random.randint(0, 6, (items_size,))

        start = datetime.now()
        result = ali.recommend(user_vec, apply_for_subscribers=True)
        end = datetime.now()
        mae_using_he[i] = evaluate(user_vec, result)
        effectiveness_using_he[i] = (end - start).microseconds

        start = datetime.now()
        result = ali.recommend(user_vec, apply_for_subscribers=False)
        end = datetime.now()
        mae_not_using_he[i] = evaluate(user_vec, result)
        effectiveness_not_using_he[i] = (end - start).microseconds

    # 打印预测结果
    print(effectiveness_using_he.mean(), effectiveness_not_using_he.mean())
    print(mae_using_he.mean(), mae_not_using_he.mean())


if __name__ == '__main__':
    # 载入数据集
    ratings_csv = os.path.abspath(os.path.join('.', 'ml-latest-small', 'ratings.csv'))
    movies_csv = os.path.abspath(os.path.join('.', 'ml-latest-small', 'movies.csv'))

    # ratings, _, hashes, _ = read_with_cache(ratings_csv, movies_csv)
    # test_lsh(ratings, hashes)

    reduce_size = (16, 64)
    ratings, _, hashes, _ = read_with_cache(ratings_csv, movies_csv, hdf5_filename='reduce.hdf5', size=reduce_size)
    test_he(ratings[:reduce_size[0], :reduce_size[1]], hashes)
