from collections import namedtuple

import numpy as np
from phe import paillier
from scipy.stats import pearsonr

EncryptedUserVec = namedtuple('EncryptedUserVec', ['ym', 'sqrt_sum_of_squares'])
HashTablesIndex = namedtuple('HashTablesIndex', ['table_index', 'bucket_index'])


class Platform:
    """
    本类抽象了一个参与到推荐计算中的平台(例如淘宝，京东)。

    本类即包含基于协同过滤和局部敏感哈希的独立推荐算法，也包括多平台参与计算的实现。本例通过在实例中注册其他平台，实时推荐时递归求推荐结果
    并汇总，模拟多平台协同合作计算。在多平台参与计算时，所有通信内容均不包含明文用户数据，并且绝不传递私钥，从而实现了用户及平台数据隐私保
    护。但由于加解密带来的开销，会使得开启多平台参与计算时推荐性能的显著下降。
    """

    def __init__(self, data=np.array([]), hashes=None, tables=4, d=8, k=None, subscribers=None) -> None:
        """
        构造计算平台
        :param data: 平台的数据矩阵
        :param hashes: 局部敏感哈希函数组所依赖的矩阵
        :param tables: 使用的哈希表数量
        :param d: 局部敏感哈希映射后的向量维度
        :param k: 平台在每次产生推荐时来自自身的结果容量
        :param subscribers: 协助计算平台列表
        """

        self.data = data
        _, cols = data.shape
        if hashes is None:
            hashes = [np.random.uniform(-1, 1, (d, cols)) for _ in range(tables)]
        else:
            assert len(hashes) > 0
            d = len(hashes[0])
            tables = len(hashes)
        self.hashes = hashes

        if k is None:
            k = len(data)
        self.k = k

        if subscribers is None:
            subscribers = []
        self.subscribers = subscribers
        self.public_key, self._private_key = paillier.generate_paillier_keypair()

        table_size = int('1' * d, 2)
        self.hash_tables = [[[] for _ in range(table_size)] for _ in range(tables)]
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
        return pearsonr(user_vec1, user_vec2)

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
        result = []
        for i, hashes_group in enumerate(self.hashes):
            index = ''
            for j, random_vec in enumerate(hashes_group):
                result = np.dot(user_vec, random_vec)
                index += 0 if result < 0 else 1
            result.append(HashTablesIndex(table_index=i, bucket_index=int(index, 2)))
        return result

    def init_lsh(self) -> None:
        """
        初始化LSH哈希表
        :return: 无
        """
        for user_vec in self.data:
            for index in self.find(user_vec):
                self.hash_tables[index.table_index][index.bucket_index].append(user_vec)

    def append_subscriber(self, platform) -> None:
        """
        添加协助计算平台
        :param platform: 协助计算平台
        :return: 无
        """
        self.subscribers.append(platform)

    def calc_similarity_for_subscriber(self, r_num, r_den):
        """
        允许其他平台通过中间值得到最终计算结果。此方法之所以存在，是因为由于使用了同态加密，协助计算平台无法独立完成最终结果的计算。
        :param r_num: 计算参数
        :param r_den: 计算参数
        :return: 最终计算结果
        """
        return self._private_key.decrypt(r_num) / self._private_key.decrypt(r_den)

    def get_similar_collection(self, user_vec):
        """
        从哈希表组中获取可能相似的所有用户参与后续计算
        :param user_vec:
        :return:
        """
        result = []
        for index in self.find(user_vec):
            result.extend(self.hash_tables[index.table_index][index.bucket_index])
        return np.array(result)

    def recommend(self, user_vec, apply_for_subscribers=False):
        """
        针对:user_vec产生实时推荐结果
        :param user_vec: 用户评分向量
        :param apply_for_subscribers: 请求其他平台参与推荐计算
        :return: 预测并填充后的用户评分向量
        """
        pass

    def participate(self, encrypted_user_vec, already_calc=set()):
        """
        参与推荐计算
        :param encrypted_user_vec: 加密的用户评分数据，内含请求计算平台的公钥
        :param already_calc: 已经参与计算的平台集合，防止同一平台重复参与本轮计算
        :return: 使用请求计算平台公钥加密后的推荐计算结果
        """
        pass
