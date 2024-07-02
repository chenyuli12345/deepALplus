import numpy as np
import torch
import torch.nn.functional as F #pytorch神经网络函数库，如激活函数和损失函数
import torch.optim as optim #包含很多pytorch优化器，如SGD，Adam等  
from torch.utils.data import DataLoader #pytorch加载数据库，用于批量加载数据集，同时提供了多种对数据集的操作方法（如数据打乱、并行、加载等）

class Strategy: #定义一个名为Strategy的类

    #Strategy类的构造函数，接收四个参数：dataset（数据集）、net（神经网络）、args_input（输入参数）、args_task（任务参数），并将这些参数保存为类的属性，以便在类的其他方法/函数中使用
    def __init__(self, dataset, net, args_input, args_task):
        self.dataset = dataset
        self.net = net
        self.args_input = args_input
        self.args_task = args_task

    def query(self, n): #接受一个参数n，
        pass #pass是一个空操作语句，表示什么都不做。这里表示query方法的实现尚未完成。这是一个占位符，用于确保代码在没有实际逻辑时仍然可以编译和运行。
    
    #定义一个名为get_labeled_count的方法，用于获取已标记数据的数量。不接受参数
    def get_labeled_count(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data() 
        #labeled_idxs是一个数组，包含已经标记好的数据的索引。
        #labeled_data是已经标记好的数据？？？
        return len(labeled_idxs) #返回已标记数据的数量
    
    #获取net对象的模型
    def get_model(self):
        return self.net.get_model()

    #更新数据集的是否有标签的布尔数组。接受两个参数：pos_idxs（正索引）和neg_idxs（负索引，可不提供，默认为None）。
    #将数据集的labeled_idxs数组（判断数据集是否标记的布尔数组）中的正索引位置的值设为True（若提供了负索引，则将负索引位置的值设为False）
    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    #训练模型。接受两个参数：data（数据集，可不提供，默认为None）和model_name（模型名称，可不提供，默认为None）。
    def train(self, data = None, model_name = None):
        if model_name == None: #若未提供模型名称
            if data == None: #若未提供数据集
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                #labeled_idxs是一个数组，包含已经标记好的数据的索引。
                #labeled_data是已经标记好的数据？？？
                self.net.train(labeled_data) #调用net对象的train方法，传入已标记数据
            else:
                self.net.train(data) #调用net对象的train方法，传入数据集
        else: #若提供了模型名称
            if model_name == 'WAAL': #若模型名称为'WAAL'
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                #labeled_idxs是一个数组，包含已经标记好的数据的索引。
                #labeled_data是已经标记好的数据？？？
                X_labeled, Y_labeled = self.dataset.get_partial_labeled_data()
                X_unlabeled, Y_unlabeled = self.dataset.get_partial_unlabeled_data()
                self.net.train(labeled_data, X_labeled, Y_labeled,X_unlabeled, Y_unlabeled)
            else:
                raise NotImplementedError

    #用模型预测数据。接受一个参数data（数据集）。
    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    #用模型预测数据的概率。接受一个参数data（数据集）。
    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings
    
    def get_grad_embeddings(self, data):
        embeddings = self.net.get_grad_embeddings(data)
        return embeddings

