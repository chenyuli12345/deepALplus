import numpy as np
import torch
import random
import os  #os模块在Python中用于与操作系统进行交互。它提供了很多方法，可以执行像读取、写入文件，查找环境变量，移动文件等操作系统级别的操作。
from torchvision import datasets 
#torchvision是一个包含了许多视觉处理工具和数据集的库，它是PyTorch的一个扩展库。
#datasets模块包含了许多预先定义好的数据集，如CIFAR, MNIST等，这些数据集常常被用于机器学习和计算机视觉的训练
from PIL import Image
#PIL是Python Imaging Library的缩写，是Python的一个图像处理库
#Image模块提供了基本的图像处理功能，如打开、操作和保存不同格式的图像。


class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler, args_task):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        self.args_task = args_task
        
        self.n_pool = len(X_train) #n_pool是训练集的大小
        self.n_test = len(X_test) #n_test是测试集的大小
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool) #生成一个长度为n_pool的全0数组，然后讲数组元素的数据类型转为布尔型，这样元素0就表示False，即n_pool个False组成的数组
        #例如，当n_pool=5时，labeled_idxs=[False False False False False]
        
    def initialize_labels(self, num): #Data类的一个方法，接收一个参数num，作用是在训练集中随机选择num个数据点，并将这些数据点标记为已标记
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool) #生成一个长度为n_pool的数组，数组元素为0到n_pool-1，代表数据池中所有可能的索引
        np.random.shuffle(tmp_idxs) #将tmp_idxs数组中的元素随机打乱
        self.labeled_idxs[tmp_idxs[:num]] = True #将（打乱后的）tmp_idxs数组中的前num个元素标记为True，即将这些数据点标记为已标记
    #最后获得的labeled_idxs是一个长度为n_pool的数组，其中num个元素为True，其余元素为False，例如[ True False False  True  True False  True False  True False]

    #按照索引idx获得第idx个未标记的数据点。输入参数是idx
    def get_unlabeled_data_by_idx(self, idx):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs] #首先生成一个0到n_pool-1的数组，~表示按位取反，互换True和False，即生成一个未标记的数据点的索引数组
        #最终获得的unlabeld_idxs是一个长度为n_pool的数组，数组的每个元素为对应labeled_idxs数组中为False的元素的索引，例如[1 2 4 6 7]
        return self.X_train[unlabeled_idxs][idx]

    #按照索引idx获得第idx个数据点以及对应的输出。输入参数是idx 
    def get_data_by_idx(self, idx):
        return self.X_train[idx], self.Y_train[idx]

    #获取新的数据，不知道这里handler是干嘛的？？？
    def get_new_data(self, X, Y):
        return self.handler(X, Y, self.args_task['transform_train'])

    #获取已标记的数据的索引，此外还要返回一个handler处理后的数据？？？
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs] #首先生成一个0到n_pool-1的数组，即生成一个标记的数据点的索引数组
        #最终获得的labeld_idxs是一个长度为n_pool的数组，数组的每个元素为对应labeled_idxs数组中为True的元素的索引，例如[0 3 5 8 9]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs], self.args_task['transform_train'])
    
    #获取未标记的数据的索引，此外还要返回一个handler处理后的数据？？？
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]#首先生成一个0到n_pool-1的数组，~表示按位取反，互换True和False，即生成一个未标记的数据点的索引数组
        #最终获得的unlabeld_idxs是一个长度为n_pool的数组，数组的每个元素为对应labeled_idxs数组中为False的元素的索引，例如[1 2 4 6 7]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs], self.args_task['transform_train'])
    
    #获取训练数据，返回两个值。第一个是对self.labeled_idxs的深拷贝（原列表改变时，返回的副本不会受到影响）；第二个是一个handler处理后的数据？？？
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, self.args_task['transform_train'])

    #获取测试数据，返回一个handler处理后的数据？？？
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test, self.args_task['transform'])
    
    #获取标记的数据，返回两个值。第一个是已标记的数据点（根据索引）；第二个是已标记的数据点的标签
    def get_partial_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs] #首先生成一个0到n_pool-1的数组，即生成一个标记的数据点的索引数组
        #最终获得的labeld_idxs是一个长度为n_pool的数组，数组的每个元素为对应labeled_idxs数组中为True的元素的索引，例如[0 3 5 8 9]
        return self.X_train[labeled_idxs], self.Y_train[labeled_idxs]
    
    #获取未标记的数据，返回两个值。第一个是未标记的数据点（根据索引）；第二个是未标记的数据点的标签
    def get_partial_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]#首先生成一个0到n_pool-1的数组，~表示按位取反，互换True和False，即生成一个未标记的数据点的索引数组
        #最终获得的unlabeld_idxs是一个长度为n_pool的数组，数组的每个元素为对应labeled_idxs数组中为False的元素的索引，例如[1 2 4 6 7]
        return self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs]

    #计算测试集的准确率，接受一个参数preds，即预测的标签。
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test
        #self.n_test是测试集的大小，self.Y_test==preds是一个长度为n_test的布尔数组，数组的每个元素为对应位置的预测标签是否与真实标签相同，sum().item()是计算布尔数组中True的个数，即预测正确的个数
    
def get_MNIST(handler, args_task):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)

def get_FashionMNIST(handler, args_task):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)

def get_EMNIST(handler, args_task):
    raw_train = datasets.EMNIST('./data/EMNIST', split = 'byclass', train=True, download=True)
    raw_test = datasets.EMNIST('./data/EMNIST', split = 'byclass', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)

def get_SVHN(handler, args_task):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data, torch.from_numpy(data_train.labels), data_test.data, torch.from_numpy(data_test.labels), handler, args_task)

def get_CIFAR10(handler, args_task):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data, torch.LongTensor(data_train.targets), data_test.data, torch.LongTensor(data_test.targets), handler, args_task)

def get_CIFAR10_imb(handler, args_task):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    X_tr = data_train.data
    Y_tr = torch.from_numpy(np.array(data_train.targets)).long()
    X_te = data_test.data
    Y_te = torch.from_numpy(np.array(data_test.targets)).long()
    ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    X_tr_imb = []
    Y_tr_imb = []
    random.seed(4666)
    for i in range(Y_tr.shape[0]):
        tmp = random.random()
        if tmp < ratio[Y_tr[i]]:
            X_tr_imb.append(X_tr[i])
            Y_tr_imb.append(Y_tr[i])
    X_tr_imb = np.array(X_tr_imb).astype(X_tr.dtype)
    Y_tr_imb = torch.LongTensor(np.array(Y_tr_imb)).type_as(Y_tr)
    return Data(X_tr_imb, Y_tr_imb, X_te, Y_te, handler, args_task)

def get_CIFAR100(handler, args_task):
    data_train = datasets.CIFAR100('./data/CIFAR100', train=True, download=True)
    data_test = datasets.CIFAR100('./data/CIFAR100', train=False, download=True)
    return Data(data_train.data, torch.LongTensor(data_train.targets), data_test.data, torch.LongTensor(data_test.targets), handler, args_task)

def get_TinyImageNet(handler, args_task):
    import cv2
    #download data from http://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip it into ./data/TinyImageNet
    # deal with training set
    Y_train_t = []
    train_img_names = []
    train_imgs = []
    
    with open('./data/TinyImageNet/tiny-imagenet-200/wnids.txt') as wnid:
        for line in wnid:
            Y_train_t.append(line.strip('\n'))
    for Y in Y_train_t:
        Y_path = './data/TinyImageNet/tiny-imagenet-200/train/' + Y + '/' + Y + '_boxes.txt'
        train_img_name = []
        with open(Y_path) as Y_p:
            for line in Y_p:
                train_img_name.append(line.strip('\n').split('\t')[0])
        train_img_names.append(train_img_name)
    train_labels = np.arange(200)
    idx = 0
    for Y in Y_train_t:
        train_img = []
        for img_name in train_img_names[idx]:
            img_path = os.path.join('./data/TinyImageNet/tiny-imagenet-200/train/', Y, 'images', img_name)
            train_img.append(cv2.imread(img_path))
        train_imgs.append(train_img)
        idx = idx + 1
    train_imgs = np.array(train_imgs)
    train_imgs = train_imgs.reshape(-1, 64, 64, 3)
    X_tr = []
    Y_tr = []
    for i in range(train_imgs.shape[0]):
        Y_tr.append(i//500)
        X_tr.append(train_imgs[i])
    #X_tr = torch.from_numpy(np.array(X_tr))
    X_tr = np.array(X_tr)
    Y_tr = torch.from_numpy(np.array(Y_tr)).long()

    #deal with testing (val) set
    Y_test_t = []
    Y_test = []
    test_img_names = []
    test_imgs = []
    with open('./data/TinyImageNet/tiny-imagenet-200/val/val_annotations.txt') as val:
        for line in val:
            test_img_names.append(line.strip('\n').split('\t')[0])
            Y_test_t.append(line.strip('\n').split('\t')[1])
    for i in range(len(Y_test_t)):
        for i_t in range(len(Y_train_t)):
            if Y_test_t[i] == Y_train_t[i_t]:
                Y_test.append(i_t)
    test_labels = np.array(Y_test)
    test_imgs = []
    for img_name in test_img_names:
        img_path = os.path.join('./data/TinyImageNet/tiny-imagenet-200/val/images', img_name)
        test_imgs.append(cv2.imread(img_path))
    test_imgs = np.array(test_imgs)
    X_te = []
    Y_te = []

    for i in range(test_imgs.shape[0]):
        X_te.append(test_imgs[i])
        Y_te.append(Y_test[i])
    #X_te = torch.from_numpy(np.array(X_te))
    X_te = np.array(X_te)
    Y_te = torch.from_numpy(np.array(Y_te)).long()
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)

def get_openml(handler, args_task, selection = 6):
    import openml
    from sklearn.preprocessing import LabelEncoder
    openml.config.apikey = '3411e20aff621cc890bf403f104ac4bc'
    openml.config.set_cache_directory('./data/openml/')
    ds = openml.datasets.get_dataset(selection)
    data = ds.get_data(target=ds.default_target_attribute)
    X = np.asarray(data[0])
    y = np.asarray(data[1])
    y = LabelEncoder().fit(y).transform(y)

    num_classes = int(max(y) + 1)
    nSamps, _ = np.shape(X)
    testSplit = .1
    inds = np.random.permutation(nSamps)
    X = X[inds]
    y = y[inds]

    split =int((1. - testSplit) * nSamps)
    while True:
        inds = np.random.permutation(split)
        if len(inds) > 50000: inds = inds[:50000]
        X_tr = X[:split]
        X_tr = X_tr[inds]
        X_tr = torch.Tensor(X_tr)

        y_tr = y[:split]
        y_tr = y_tr[inds]
        Y_tr = torch.Tensor(y_tr).long()

        X_te = torch.Tensor(X[split:])
        Y_te = torch.Tensor(y[split:]).long()

        if len(np.unique(Y_tr)) == num_classes: break
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)

def get_BreakHis(handler, args_task):
    # download data from https://www.kaggle.com/datasets/ambarish/breakhis and unzip it in data/BreakHis/
    data_dir = './data/BreakHis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'
    data = datasets.ImageFolder(root = data_dir, transform = None).imgs
    train_ratio = 0.7
    test_ratio = 0.3
    data_idx = list(range(len(data)))
    random.shuffle(data_idx)
    train_idx = data_idx[:int(len(data)*train_ratio)]
    test_idx = data_idx[int(len(data)*train_ratio):]
    X_tr = [np.array(Image.open(data[i][0])) for i in train_idx]
    Y_tr = [data[i][1] for i in train_idx]
    X_te = [np.array(Image.open(data[i][0])) for i in test_idx]
    Y_te = [data[i][1] for i in test_idx]
    X_tr = np.array(X_tr, dtype=object)
    X_te = np.array(X_te, dtype=object)
    Y_tr = torch.from_numpy(np.array(Y_tr))
    Y_te = torch.from_numpy(np.array(Y_te))
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)

def get_PneumoniaMNIST(handler, args_task):
    # download data from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia and unzip it in data/PhwumniaMNIST/
    import cv2

    data_train_dir = './data/PneumoniaMNIST/chest_xray/train/'
    data_test_dir = './data/PneumoniaMNIST/chest_xray/test/'
    assert os.path.exists(data_train_dir)
    assert os.path.exists(data_test_dir)

    #train data
    train_imgs_path_0 = [data_train_dir+'NORMAL/'+f for f in os.listdir(data_train_dir+'/NORMAL/')]
    train_imgs_path_1 = [data_train_dir+'PNEUMONIA/'+f for f in os.listdir(data_train_dir+'/PNEUMONIA/')]
    train_imgs_0 = []
    train_imgs_1 = []
    for p in train_imgs_path_0:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        train_imgs_0.append(im)
    for p in train_imgs_path_1:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        train_imgs_1.append(im)
    train_labels_0 = np.zeros(len(train_imgs_0))
    train_labels_1 = np.ones(len(train_imgs_1))
    X_tr = []
    Y_tr = []
    train_imgs = train_imgs_0 + train_imgs_1
    train_labels = np.concatenate((train_labels_0, train_labels_1))
    idx_train = list(range(len(train_imgs)))
    random.seed(4666)
    random.shuffle(idx_train)
    X_tr = [train_imgs[i] for i in idx_train]
    Y_tr = [train_labels[i] for i in idx_train]
    X_tr = np.array(X_tr)
    Y_tr = torch.from_numpy(np.array(Y_tr)).long()

    #test data
    test_imgs_path_0 = [data_test_dir+'NORMAL/'+f for f in os.listdir(data_test_dir+'/NORMAL/')]
    test_imgs_path_1 = [data_test_dir+'PNEUMONIA/'+f for f in os.listdir(data_test_dir+'/PNEUMONIA/')]
    test_imgs_0 = []
    test_imgs_1 = []
    for p in test_imgs_path_0:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        test_imgs_0.append(im)
    for p in test_imgs_path_1:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        test_imgs_1.append(im)
    test_labels_0 = np.zeros(len(test_imgs_0))
    test_labels_1 = np.ones(len(test_imgs_1))
    X_te = []
    Y_te = []
    test_imgs = test_imgs_0 + test_imgs_1
    test_labels = np.concatenate((test_labels_0, test_labels_1))
    idx_test = list(range(len(test_imgs)))
    random.seed(4666)
    random.shuffle(idx_test)
    X_te = [test_imgs[i] for i in idx_test]
    Y_te = [test_labels[i] for i in idx_test]
    X_te = np.array(X_te)
    Y_te = torch.from_numpy(np.array(Y_te)).long()

    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)


def get_waterbirds(handler, args_task):
    import wilds
    from torchvision import transforms
    dataset = wilds.get_dataset(dataset='waterbirds', root_dir='./data/waterbirds', download='True')
    trans = transforms.Compose([transforms.Resize([255,255])])
    train = dataset.get_subset(split = 'train',transform = trans)
    test = dataset.get_subset(split = 'test', transform = trans)

    len_train = train.metadata_array.shape[0]
    len_test = test.metadata_array.shape[0]
    X_tr = []
    Y_tr = []
    X_te = []
    Y_te = []

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    f = open('waterbirds.txt', 'w')

    for i in range(len_train):
        x,y,meta = train.__getitem__(i)
        img = np.array(x)
        X_tr.append(img)
        Y_tr.append(y)

    for i in range(len_test):
        x,y, meta = test.__getitem__(i)
        img = np.array(x)

        X_te.append(img)
        Y_te.append(y)
        if meta[0] == 0 and meta[1] == 0:
            f.writelines('1') #landbird_background:land
            f.writelines('\n')
            count1 = count1 + 1
        elif meta[0] == 1 and meta[1] == 0:
            f.writelines('2') #landbird_background:water
            count2 = count2 + 1
            f.writelines('\n')
        elif meta[0] == 0 and meta[1] == 1:
            f.writelines('3') #waterbird_background:land
            f.writelines('\n')
            count3 = count3 + 1
        elif meta[0] == 1 and meta[1] == 1:
            f.writelines('4') #waterbird_background:water
            f.writelines('\n')
            count4 = count4 + 1
        else:
            raise NotImplementedError    
    f.close()

    Y_tr = torch.tensor(Y_tr)
    Y_te = torch.tensor(Y_te)
    X_tr = np.array(X_tr)
    X_te = np.array(X_te)
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)














