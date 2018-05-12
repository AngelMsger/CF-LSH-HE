from collections import namedtuple

import numpy as np
from phe import paillier
from scipy.stats import pearsonr

from utils import log_duration

EncryptedUserVec = namedtuple('EncryptedUserVec', ['ym', 'sqrt_sum_of_squares'])
HashTablesIndex = namedtuple('HashTablesIndex', ['table_index', 'bucket_index'])


class Platform:
    """
    本类抽象了一个参与到推荐计算中的平台(例如淘宝，京东)。

    本类即包含基于协同过滤和局部敏感哈希的独立推荐算法，也包括多平台参与计算的实现。本例通过在实例中注册其他平台，实时推荐时递归求推荐结果
    并汇总，模拟多平台协同合作计算。在多平台参与计算时，所有通信内容均不包含明文用户数据，并且绝不传递私钥，从而实现了用户及平台数据隐私保
    护。但由于加解密带来的开销，会使得开启多平台参与计算时推荐性能的显著下降。
    """

    def __init__(self, name, hashes, data=np.array([]), tables=4, d=8, k=None, subscribers=None) -> None:
        """
        构造计算平台
        :param name: 平台ID
        :param hashes: 局部敏感哈希函数组所依赖的矩阵组，不同平台必须传入相同矩阵组才能正常参与协同工作
        :param data: 平台的数据矩阵
        :param tables: 使用的哈希表数量
        :param d: 局部敏感哈希映射后的向量维度
        :param k: 平台在每次产生推荐时来自自身的结果容量
        :param subscribers: 协助计算平台列表
        """

        self.id = name
        self.data = data
        _, cols = data.shape

        assert len(hashes) > 0
        self.hashes = hashes
        d = len(hashes[0])
        tables = len(hashes)

        if k is None:
            k = len(data)
        self.k = k

        if subscribers is None:
            subscribers = []
        self.subscribers = subscribers
        self.public_key, self._private_key = paillier.generate_paillier_keypair()

        table_size = int('1' * d, 2) + 1
        self.hash_tables = [[set() for _ in range(table_size)] for _ in range(tables)]
        self.init_lsh()

    @staticmethod
    def calc_similarity(user_vec1, user_vec2):
        """
        利用皮尔逊相关系数计算向量间相似度
        :param user_vec1: 用户向量1
        :param user_vec2: 用户向量2
        :return: 相似度计算结果
        """
        assert len(user_vec1) == len(user_vec2) != 0
        return pearsonr(user_vec1, user_vec2)[0]

    @staticmethod
    def calc_similarity_with_encrypted_data(user_vec, encrypted_user_vec, platform):
        """
        为其他平台利用皮尔逊相关系数计算明文向量与加密向量间相似度
        :param user_vec: 明文用户向量
        :param encrypted_user_vec: 加密的用户向量
        :param platform: 请求计算的平台
        :return: 相似度计算结果
        """
        assert len(user_vec) == len(encrypted_user_vec.ym) != 0
        x = np.asarray(user_vec)
        xm = x - x.mean()
        r_num = np.dot(xm, encrypted_user_vec.ym)
        r_den = np.sqrt(np.add.reduce(xm ** 2)) * encrypted_user_vec.sqrt_sum_of_squares
        return platform.calc_similarity_for_subscriber(r_num, r_den)

    def find(self, user_vec):
        """
        在哈希表中查找用户向量索引
        :param user_vec: 用户向量
        :return: 结果列表，包含多个由哈希表编号和桶编号组成的对
        """
        indexes = []
        for i, hashes_group in enumerate(self.hashes):
            index = ''
            for j, random_vec in enumerate(hashes_group):
                result = np.dot(user_vec, random_vec)
                index += '0' if result < 0 else '1'
            indexes.append(HashTablesIndex(table_index=i, bucket_index=int(index, 2)))
        return indexes

    def init_lsh(self) -> None:
        """
        初始化LSH哈希表
        :return: 无
        """
        for i, user_vec in enumerate(self.data):
            for index in self.find(user_vec):
                self.hash_tables[index.table_index][index.bucket_index].add(i)

    def append_subscriber(self, platform) -> None:
        """
        添加协助计算平台
        :param platform: 协助计算平台
        :return: 无
        """
        assert self.hashes == platform.hashes
        self.subscribers.append(platform)

    def calc_similarity_for_subscriber(self, r_num, r_den):
        """
        允许其他平台通过中间值得到最终计算结果。此方法之所以存在，是因为由于使用了同态加密，协助计算平台无法独立完成最终结果的计算。
        :param r_num: 计算参数
        :param r_den: 计算参数
        :return: 最终计算结果
        """
        return self._private_key.decrypt(r_num) / self._private_key.decrypt(r_den)

    def get_similar_collection(self, hash_table_indexes):
        """
        从哈希表组中获取可能相似的所有用户参与后续计算
        :param hash_table_indexes:
        :return:
        """
        result = set()
        for index in hash_table_indexes:
            result |= self.hash_tables[index.table_index][index.bucket_index]
        return np.array([self.data[i] for i in result])

    @log_duration
    def recommend(self, user_vec, use_lsh=True, apply_for_subscribers=False):
        """
        针对:user_vec产生实时推荐结果
        :param user_vec: 用户评分向量
        :param use_lsh: 启用局部敏感哈希
        :param apply_for_subscribers: 请求其他平台参与推荐计算
        :return: 预测并填充后的用户评分向量
        """
        user_vec = np.asarray(user_vec)
        hash_table_indexes = self.find(user_vec)

        if use_lsh:
            collection = self.get_similar_collection(hash_table_indexes)
        else:
            collection = self.data

        users_count, items_count = collection.shape

        similarities = np.empty((users_count,))
        for i in range(users_count):
            evaluated = np.logical_and(user_vec > 0, collection[i] > 0)
            similarities[i] = self.calc_similarity(user_vec[evaluated], collection[i][evaluated])
        most_similar_indexes = similarities.argsort()[0 - self.k:]

        sum_of_similarity, sum_of_ratings_with_weights = 0, np.zeros((items_count,))
        for index in most_similar_indexes:
            sum_of_similarity += similarities[index]
            sum_of_ratings_with_weights += similarities[index] * collection[index]

        if apply_for_subscribers:
            effective_index = user_vec > 0
            effective_user_vec = user_vec[effective_index]
            encrypted_user_vec = [self.public_key.encrypt(i.item()) for i in effective_user_vec]
            for platform in self.subscribers:
                x, y = platform.participate(encrypted_user_vec, effective_index, hash_table_indexes, self)
                sum_of_ratings_with_weights += np.array([self._private_key.decrypt(i) for i in x])
                sum_of_similarity += self._private_key.decrypt(y)

        return sum_of_ratings_with_weights / sum_of_similarity

    def get_effective_encrypted_user_vec(self, encrypted_user_vec):
        """
        为参与计算平台提供可以被计算的加密数据
        :param encrypted_user_vec:
        :return:
        """
        user_vec = np.array([self._private_key.decrypt(i) for i in encrypted_user_vec])
        ym = user_vec - user_vec.mean()
        sqrt_sum_of_squares = np.sqrt(np.add.reduce(ym ** 2))
        return EncryptedUserVec(ym=self.public_key.encrypt(ym),
                                sqrt_sum_of_squares=self.public_key.encrypt(sqrt_sum_of_squares))

    def encrypted_data_argsort(self, vector):
        """
        为参与计算平台提供加密数据排序后的索引
        :param vector: 需要排序的加密数据向量
        :return: 排序后的索引值
        """
        return np.array([self._private_key.decrypt(i) for i in vector]).argsort()

    def participate(self, encrypted_user_vec, effective_index, hash_table_indexes, platform):
        """
        参与推荐计算
        :param encrypted_user_vec: 加密的用户评分数据，内含请求计算平台的公钥
        :param effective_index: 传入的有效维度对应的索引，用来告知参与平台仅需考虑用户向量的这些维度
        :param hash_table_indexes: 用户对应的哈希索引，不涉及用户数据本身，不会暴露数据隐私
        :param platform: 请求计算的平台
        :return: 使用请求计算平台公钥加密后的推荐计算结果
        """
        collection = self.get_similar_collection(hash_table_indexes)
        similarities = np.empty((len(collection),))
        for i, user_vec in enumerate(collection):
            effective_user_vec = user_vec[effective_index]
            indexes = np.where(effective_user_vec > 0)
            effective_index = set([int(i) for i in indexes])
            effective_user_vec = effective_user_vec[effective_user_vec > 0]
            encrypted_user_vec = [vec for i, vec in enumerate(encrypted_user_vec) if i in effective_index]
            similarities[i] = self.calc_similarity_with_encrypted_data(
                effective_user_vec, self.get_effective_encrypted_user_vec(encrypted_user_vec), platform)
        most_similar_indexes = platform.encrypted_data_argsort(similarities)[0 - self.k:]

        _, items_count = self.data.shape
        sum_of_similarity, sum_of_ratings_with_weights = 0, [0 for _ in range(items_count)]
        for index in most_similar_indexes:
            sum_of_similarity += similarities[index]
            for i, value in enumerate(collection[index]):
                sum_of_ratings_with_weights[i] += similarities[index] * value
        return sum_of_ratings_with_weights, sum_of_similarity
