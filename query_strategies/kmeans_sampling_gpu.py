import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans #sklearn是一个广泛使用的机器学习库，提供了各种算法和工具用于数据挖掘和分析。Kmeans是聚类算法，用于将数据分成预先指定数量的簇
import faiss #一种高效的相似性搜索和密集向量聚类库，适用于在大规模数据中进行高维向量的搜索和聚类操作

class KMeansSamplingGPU(Strategy):
    def __init__(self, dataset, net, args_input, args_task): #聚类的构造函数，接受四个参数：dataset（数据集）、net（神经网络）、args_input（输入参数）、args_task（任务参数），并将这些参数保存为类的属性，以便在类的其他方法/函数中使用
        super(KMeansSamplingGPU, self).__init__(dataset, net, args_input, args_task) #调用父类Strategy的构造函数，它初始化了这个类的基础结构

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        embeddings = self.get_embeddings(unlabeled_data).numpy()
        cluster_learner = FaissKmeans(n_clusters = n, gpu = True)
        cluster_learner.fit(embeddings)
        dis, q_idxs = cluster_learner.predict(embeddings)
        q_idxs = q_idxs.T[0]
        
        return unlabeled_idxs[q_idxs]


class FaissKmeans:
    def __init__(self, n_clusters=8, gpu=True, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.gpu = gpu

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init,
                                   gpu = self.gpu)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        D, I = self.kmeans.index.search(X.astype(np.float32), 1)
        return D, I