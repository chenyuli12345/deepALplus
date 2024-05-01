#最大熵方法
import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy): #继承Strategy类
    def __init__(self, dataset, net, args_input, args_task): #一种特殊的方法，在创建类的实例时被调用。这个方法中将传入的参数（数据集、网络、输入参数和任务参数）保存为类的属性
        super(EntropySampling, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        #从数据集中获取未标记的数据
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        #使用网络预测这些数据的概率
        probs = self.predict_prob(unlabeled_data)
        #计算数据的不确定性：概率的对数和概率的乘积
        log_probs = torch.log(probs)
        uncertainties = (probs*log_probs).sum(1)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
