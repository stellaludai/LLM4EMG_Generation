import torch
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import scipy.io as scio
from torch.utils import data
from sklearn.model_selection import train_test_split

"""
这是一个基本的框架
"""

"""
读取矩阵
返回改矩阵的emg和restimulus
"""
def LoadMat(path):
    mat = scio.loadmat(path)
    emg = mat['emg']
    label = mat['restimulus']
    return emg, label


"""
切割数据集
"""
def cut(emg, label):
    ans_emg = []
    ans_label = []
    i = 0
    while(i < len(label)):
        if(i + 199 >= len(label) - 1):
            break
        if(label[i] == label[i+199]):
            start = i
            end = i + 200
            temp_emg = emg[start : end]
            temp_label = label[i]
            ans_emg.append(temp_emg)
            ans_label.append(temp_label)
            i += 50
        else:
            i += 50
    return ans_emg, ans_label

"""
统计不同标签的个数
"""
def count_unique_labels(labels):
    label_count = {}  # 用字典来存储不同标签及其对应的个数

    for label_list in labels:
        label = label_list[0]  # 提取单个元素的标签
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1

    for label, count in label_count.items():
        print(f"标签 {label} 的个数：{count}")


E1_emg, E1_label = LoadMat('D:/Desktop/data/ninapro/db1/DB1_s1/S1_A1_E1.mat')
E2_emg, E2_label = LoadMat('D:/Desktop/data/ninapro/db1/DB1_s1/S1_A1_E2.mat')
E3_emg, E3_label = LoadMat('D:/Desktop/data/ninapro/db1/DB1_s1/S1_A1_E3.mat')

emgs, labels = cut(E1_emg, E1_label)
E2_emg, E2_label = cut(E2_emg, E2_label)
E3_emg, E3_label = cut(E3_emg, E3_label)
emgs.extend(E2_emg)
emgs.extend(E3_emg)
labels.extend(E2_label)
labels.extend(E3_label)

import random

# 找到所有标签为 0 的样本的索引
label_0_indices = [i for i, label in enumerate(labels) if label == 0]

# 随机选择 100 个标签为 0 的样本的索引
selected_indices = random.sample(label_0_indices, min(3600, len(label_0_indices)))

# 保留不在选定索引中的样本和标签
emgs = [emgs[i] for i in range(len(emgs)) if i not in selected_indices]
labels = [labels[i] for i in range(len(labels)) if i not in selected_indices]



count_unique_labels(labels)  # 统计不同标签的数据样本量

class EMG_dataset(data.Dataset):
    def __init__(self, emgs, labels):
        self.emgs = emgs
        self.labels = labels

    def __getitem__(self, index):
        emg = emgs[index]
        label = labels[index]
        return emg, label

    def __len__(self):
        return len(self.emgs)


train_emgs, test_emgs, train_labels, test_labels = train_test_split(emgs, labels, test_size=0.2, random_state=42)

# 创建训练集的 EMG_dataset 和 DataLoader
train_dataset = EMG_dataset(train_emgs, train_labels)
train_dataloader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 创建测试集的 EMG_dataset 和 DataLoader
test_dataset = EMG_dataset(test_emgs, test_labels)
test_dataloader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

"""
创建模型
"""

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_1 = nn.Linear(2000, 1000)  # 1000为超参数，可以自己选择
        self.linear_2 = nn.Linear(1000, 648)
        self.linear_3 = nn.Linear(648, 328)
        self.linear_4 = nn.Linear(328, 168)
        self.linear_5 = nn.Linear(168, 24)  # 输出为13个类别

    def forward(self, input):
        x = input.view(-1, 200 * 10)
        x = x.to(self.linear_1.weight.dtype)
        x = torch.relu(self.linear_1(x))
        x = torch.relu(self.linear_2(x))
        x = torch.relu(self.linear_3(x))
        x = torch.relu(self.linear_4(x))
        logits = self.linear_5(x)  # 未激活的输出叫做logits
        return logits

"""
初始化损失函数
"""
loss_fn = torch.nn.CrossEntropyLoss()

"""
优化:根据计算得到的损失，调整模型参数
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
model = Model().to(device)  # 初始化模型
opt = torch.optim.SGD(model.parameters(), lr=0.001)

"""
训练函数:对所有数据训练一次
"""
def train(dl, model, loss_fn, optimizer):
    size = len(dl.dataset)  # 获取当前数据集的大小
    num_batches = len(dl)  # 获取总批次数


    train_loss, correct = 0, 0

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        y = y.view(-1)

        pre = model(x)
        loss = loss_fn(pre, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            correct += (pre.argmax(1) == y).type(torch.float).sum().item()
            train_loss += loss.item()

    correct /= size
    train_loss /= num_batches

    return correct, train_loss

def test(test_dl, model, loss_fn):
    size = len(test_dl.dataset)  # 获取当前数据集的大小
    num_batches = len(test_dl)  # 获取总批次数

    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            y = y.view(-1)
            pre = model(x)
            loss = loss_fn(pre, y)
            correct += (pre.argmax(1) == y).type(torch.float).sum().item()
            test_loss += loss.item()
        correct /= size
        test_loss /= num_batches

        return correct, test_loss


epochs = 2000
train_loss = []
train_acc = []  # 正确率
test_loss = []
test_acc = []


for epoch in range(epochs):
    epoch_train_acc, epoch_train_loss = train(train_dataloader, model, loss_fn, opt)
    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)

    epoch_test_acc, epoch_test_loss = test(test_dataloader, model, loss_fn)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)

    template = ('epoch:{:2d}, train_Loss:{:.5f}, train_acc:{:.1f}, test_Loss:{:.5f}, test_acc:{:.1f}')
    print(template.format(epoch, epoch_train_loss, epoch_train_acc*100, epoch_test_loss, epoch_test_acc*100))