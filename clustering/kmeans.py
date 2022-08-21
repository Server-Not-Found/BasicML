import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class KMeans:
    def __init__(self,data,k_num):
        self.data = data
        self.k_num = k_num

    def train(self,max_iter):
        """
        KMeans算法训练过程
        :param self:
        :param max_iter:
        :return:
            centroids:中心点矩阵
            closest_centroids_ids:每个样本点特征最近的中心点分布
        """
        # 1.随机选择K个中心点
        centroids = KMeans.centroids_init(self.data,self.k_num)
        num_examples = self.data.shape[0]
        # 2.初始化每个样本点离得最近的中心点
        closest_centroids_ids = np.empty((num_examples,1))
        for i in range(max_iter):
            # 计算每个样本点离得最近的中心点
            closest_centroids_ids = KMeans.find_closest_centroids(self.data,centroids)
            # 3.更新中心点位置
            centroids,closest_centroids_ids = KMeans.centroids_update(self.data,closest_centroids_ids,self.k_num)
        return centroids, closest_centroids_ids

    @staticmethod
    def centroids_init(data,k_num):
        """
        初始化中心点
        :return: 中心点列表
        """
        num_examples = data.shape[0]
        random_ids = np.random.permutation(num_examples)    # 对样本序列进行随机排序
        centroids = data[random_ids[:k_num],:]
        return centroids

    @staticmethod
    def find_closest_centroids(data,centroids):
        """
        :param data:
        :param centroids:
        :return: 每个样本对应的所属簇的下标列表
        """
        num_examples = data.shape[0]   # 样本个数
        num_centroids = centroids.shape[0]   # 中心点个数
        closest_centroids_ids = np.zeros((num_examples,1))  # 每个样本的最近距离
        for example_index in range(num_examples):
            distance = np.zeros((num_centroids,1))
            for centroids_index in range(num_centroids):
                distance_diff = data[example_index,:] - centroids[centroids_index,:]
                distance[centroids_index] = np.sum(distance_diff**2) # distance是所有特征的综合距离
            closest_centroids_ids[example_index] = np.argmin(distance)  # 得到最小距离对应的簇
        return closest_centroids_ids

    @staticmethod
    def centroids_update(data,closest_centroids_ids,k_num):
        num_features = data.shape[1]    # 特征个数
        centroids = np.zeros((k_num,num_features))  # 中心点的坐标列表
        # 遍历K个簇，找到本次迭代属于该簇的样本点
        for cluster in range(k_num):
            closest_ids = closest_centroids_ids == cluster
            centroids[cluster] = np.mean(data[closest_ids.flatten(),:],axis=0) # axis=0对各列求均值
        return centroids,closest_centroids_ids
