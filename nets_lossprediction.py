import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #pytorch神经网络函数库，如激活函数和损失函数
import torch.optim as optim #包含很多pytorch优化器，如SGD，Adam等  
from torch.utils.data import DataLoader #pytorch加载数据库，用于批量加载数据集，同时提供了多种对数据集的操作方法（如数据打乱、并行、加载等）
import torchvision.models as models
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm

# LossPredictionLoss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], batch size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already haved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss

#网络的类
class Net_LPL:
    def __init__(self, net, params, device, net_lpl): #类的构造函数，接收四个参数：net（神经网络）、params（参数）、device（设备）、net_lpl（损失预测网络），用于初始化对象
        #将构造函数接受的参数赋值给类的实例变量
        self.net = net
        self.params = params
        self.device = device
        self.net_lpl = net_lpl
        
    def train(self, data, weight = 1.0, margin = 1.0 , lpl_epoch = 20): #类的一个方法train，用于训练模型。接收三个参数：data（数据集）、weight（权重，默认为1.0）、margin（边界，默认为1.0）、lpl_epoch（损失预测网络的训练轮数，默认为20）
        n_epoch = self.params['n_epoch'] #从self.params字典中获取正常训练的迭代次数n_epoch
        n_epoch = lpl_epoch + self.params['n_epoch'] #将损失预测训练的迭代次数（来自于输入参数）加到正常训练的迭代次数上
        epoch_loss = lpl_epoch #将损失预测训练的迭代次数赋值给epoch_loss

        dim = data.X.shape[1:] #获取数据集的维度

        #创建一个网络模型实例self.clf，并将其移动到指定设备上（来自于参数）
        self.clf = self.net(dim = dim, pretrained = self.params['pretrained'], num_classes = self.params['num_class']).to(self.device)
        
        #将损失预测网络模型self.clf_lpl移动到指定设备上（来自于参数）
        self.clf_lpl = self.net_lpl.to(self.device)
        #self.clf.train()

        if self.params['optimizer'] == 'Adam': #如果参数字典中的优化器为Adam
            optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args']) #创建一个优化器，括号内为要优化的参数，使用Adam优化方法。第一组参数为self.clf的参数，第二组参数为self.params字典中的优化器参数（解包字典，将其作为关键字传递给优化器）
        elif self.params['optimizer'] == 'SGD': #如果参数字典中的优化器为SGD
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args']) #创建一个优化器，括号内为要优化的参数，使用SGD优化方法。第一组参数为self.clf的参数，第二组参数为self.params字典中的优化器参数（解包字典，将其作为关键字传递给优化器）
        else: #如果参数字典中的优化器不是Adam或SGD
            raise NotImplementedError #抛出一个NotImplementedError异常，表示代码遇到了一个未实现的优化器类型
        optimizer_lpl = optim.Adam(self.clf_lpl.parameters(), lr = 0.01) #创建另一个优化器实例，用于优化损失预测网络self.clf_lpl的参数，使用Adam优化方法，学习率为0.01

        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args']) #使用DataLoader类创建一个数据加载器实例。这个加载器用于迭代地加载数据集。data变量代表了要加载的数据集，shuffle=True参数表示在每个epoch开始时，数据将被随机打乱，以帮助模型更好地学习
        self.clf.train() #将神经网络设置为训练模式而不是评估模式
        self.clf_lpl.train() #将损失预测网络设置为训练模式而不是评估模式
        for epoch in tqdm(range(1, n_epoch+1), ncols=100): #循环迭代n_epoch次，每次迭代的索引为epoch。ncols表示将进度条的宽度设置为100
            for batch_idx, (x, y, idxs) in enumerate(loader): #循环遍历数据加载器（loader）返回的每个批次的数据。每个批次包含输入数据x，标签y，和索引idxs
                x, y = x.to(self.device), y.to(self.device) #将输入数据x和标签y移动到指定设备上（来自于参数）
                optimizer.zero_grad() #清除之前的梯度，为新的梯度计算做准备
                optimizer_lpl.zero_grad() #清除之前的梯度，为新的梯度计算做准备
                out, feature = self.clf(x) #将输入数据x传递给神经网络self.clf（前向传播），得到输出out和特征feature
                out, e1 = self.clf(x) #？？重复执行上一步，将输入数据x传递给神经网络self.clf（前向传播），得到输出out和特征feature
                cross_ent = nn.CrossEntropyLoss(reduction='none')
                target_loss = cross_ent(out,y)
                if epoch >= epoch_loss:
                    feature[0] = feature[0].detach()
                    feature[1] = feature[1].detach()
                    feature[2] = feature[2].detach()
                    feature[3] = feature[3].detach()
                pred_loss = self.clf_lpl(feature)
                pred_loss = pred_loss.view(pred_loss.size(0))

                backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                module_loss = LossPredLoss(pred_loss, target_loss, margin)
                loss = backbone_loss + weight * module_loss
                loss.backward()
                optimizer.step()
                optimizer_lpl.step()

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds
    
    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs
    
    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs
    
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs
    
    def get_model(self):
        return self.clf

    def get_embeddings(self, data):
        self.clf.eval() #将模型设为评估模式
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings
    
    def get_grad_embeddings(self, data):
        self.clf.eval()
        embDim = self.clf.get_embedding_dim()
        nLab = self.params['num_class']
        embeddings = np.zeros([len(data), embDim * nLab])

        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embeddings[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c]) * -1.0
                        else:
                            embeddings[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c]) * -1.0

        return embeddings
        
class MNIST_Net_LPL(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		
		self.feature0 = nn.Sequential(*list(resnet18.children())[0:3])
		self.feature1 = nn.Sequential(*list(resnet18.children())[4])
		self.feature2 = nn.Sequential(*list(resnet18.children())[5])
		self.feature3 = nn.Sequential(*list(resnet18.children())[6]) 
		self.feature4 = nn.Sequential(*list(resnet18.children())[7])
		self.feature5 = nn.Sequential(*list(resnet18.children())[8:9])
		self.conv = nn.Conv2d(1, 3, kernel_size = 1)
		self.classifier = nn.Linear(resnet18.fc.in_features,num_classes)
		self.dim = resnet18.fc.in_features
		
	def forward(self, x):
		x = self.conv(x)
		x0 = self.feature0(x)
		x1 = self.feature1(x0)
		x2 = self.feature2(x1)
		x3 = self.feature3(x2)
		x4 = self.feature4(x3)
		x5 = self.feature5(x4)
		output = x5.view(x5.size(0), -1)
		output = self.classifier(output)
		return output, [x1,x2,x3,x4]
	
	
	def get_embedding_dim(self):
		return self.dim

class CIFAR10_Net_LPL(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		self.feature0 = nn.Sequential(*list(resnet18.children())[0:3])
		self.feature1 = nn.Sequential(*list(resnet18.children())[4])
		self.feature2 = nn.Sequential(*list(resnet18.children())[5])
		self.feature3 = nn.Sequential(*list(resnet18.children())[6]) 
		self.feature4 = nn.Sequential(*list(resnet18.children())[7])
		self.feature5 = nn.Sequential(*list(resnet18.children())[8:9])
		self.classifier = nn.Linear(512, num_classes)

		self.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.dim = resnet18.fc.in_features

		
	
	def forward(self, x):

		x0 = self.feature0(x)
		x1 = self.feature1(x0)
		x2 = self.feature2(x1)
		x3 = self.feature3(x2)
		x4 = self.feature4(x3)
		x5 = self.feature5(x4)
		output = x5.view(x5.size(0), -1)
		output = self.classifier(output)
		return output, [x1,x2,x3,x4]
	
	def get_embedding_dim(self):
		return self.dim

class openml_Net(nn.Module):
    def __init__(self, dim = 28 * 28, embSize=256, pretrained=False, num_classes = 10):
        super(openml_Net, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, num_classes)
    
    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        out = self.lm2(emb)
        return out, emb
    
    def get_embedding_dim(self):
        return self.embSize

class PneumoniaMNIST_Net_LPL(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		self.feature0 = nn.Sequential(*list(resnet18.children())[0:3])
		self.feature1 = nn.Sequential(*list(resnet18.children())[4])
		self.feature2 = nn.Sequential(*list(resnet18.children())[5])
		self.feature3 = nn.Sequential(*list(resnet18.children())[6]) 
		self.feature4 = nn.Sequential(*list(resnet18.children())[7])
		self.feature5 = nn.Sequential(*list(resnet18.children())[8:9])
		self.classifier = nn.Linear(512, num_classes)

		self.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.feature0[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.dim = resnet18.fc.in_features
	
	
	def forward(self, x):

		x0 = self.feature0(x)
		x1 = self.feature1(x0)
		x2 = self.feature2(x1)
		x3 = self.feature3(x2)
		x4 = self.feature4(x3)
		x5 = self.feature5(x4)
		output = x5.view(x5.size(0), -1)
		output = self.classifier(output)
		return output, [x1,x2,x3,x4]
	
	def get_embedding_dim(self):
		return self.dim

class waterbirds_Net_LPL(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		self.feature0 = nn.Sequential(*list(resnet18.children())[0:3])
		self.feature1 = nn.Sequential(*list(resnet18.children())[4])
		self.feature2 = nn.Sequential(*list(resnet18.children())[5])
		self.feature3 = nn.Sequential(*list(resnet18.children())[6]) 
		self.feature4 = nn.Sequential(*list(resnet18.children())[7])
		self.feature5 = nn.Sequential(*list(resnet18.children())[8:9])
		self.classifier = nn.Linear(resnet18.fc.in_features, num_classes)
		self.dim = resnet18.fc.in_features
		
	
	def forward(self, x):
		x0 = self.feature0(x)
		x1 = self.feature1(x0)
		x2 = self.feature2(x1)
		x3 = self.feature3(x2)
		x4 = self.feature4(x3)
		x5 = self.feature5(x4)
		output = x5.view(x5.size(0), -1)
		output = self.classifier(output)
		return output, [x1,x2,x3,x4]
	
	def get_embedding_dim(self):
		return self.dim


def get_lossnet(name):
	if name == 'PneumoniaMNIST':
		return LossNet(feature_sizes=[224, 112, 56, 28], num_channels=[64, 128, 256, 512], interm_dim=128)
	elif 'MNIST' in name:
		return LossNet(feature_sizes=[14, 7, 4, 2], num_channels=[64, 128, 256, 512], interm_dim=128) 
	elif 'CIFAR' in name:
		return LossNet(feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128)
	elif 'ImageNet' in name:
		return LossNet(feature_sizes=[64, 32, 16, 8], num_channels=[64, 128, 256, 512], interm_dim=128)
	elif 'BreakHis' in name:
		return LossNet(feature_sizes=[224, 112, 56, 28], num_channels=[64, 128, 256, 512], interm_dim=128)
	elif 'waterbirds' in name:
		return LossNet(feature_sizes=[128, 64, 32, 16], num_channels=[64, 128, 256, 512], interm_dim=128)
	else:
		raise NotImplementedError

class LossNet(nn.Module):
	def __init__(self, feature_sizes=[28, 14, 7, 4], num_channels=[64, 128, 256, 512], interm_dim=128):
		super(LossNet, self).__init__()
		
		self.GAP1 = nn.AvgPool2d(feature_sizes[0])
		self.GAP2 = nn.AvgPool2d(feature_sizes[1])
		self.GAP3 = nn.AvgPool2d(feature_sizes[2])
		self.GAP4 = nn.AvgPool2d(feature_sizes[3])

		self.FC1 = nn.Linear(num_channels[0], interm_dim)
		self.FC2 = nn.Linear(num_channels[1], interm_dim)
		self.FC3 = nn.Linear(num_channels[2], interm_dim)
		self.FC4 = nn.Linear(num_channels[3], interm_dim)

		self.linear = nn.Linear(4 * interm_dim, 1)
	
	def forward(self, features):

		out1 = self.GAP1(features[0])
		out1 = out1.view(out1.size(0), -1)
		out1 = F.relu(self.FC1(out1))

		out2 = self.GAP2(features[1])
		out2 = out2.view(out2.size(0), -1)
		out2 = F.relu(self.FC2(out2))

		out3 = self.GAP3(features[2])
		out3 = out3.view(out3.size(0), -1)
		out3 = F.relu(self.FC3(out3))

		out4 = self.GAP4(features[3])
		out4 = out4.view(out4.size(0), -1)
		out4 = F.relu(self.FC4(out4))

		out = self.linear(torch.cat((out1, out2, out3, out4), 1))
		return out
