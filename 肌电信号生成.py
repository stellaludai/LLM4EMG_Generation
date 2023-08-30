import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import scipy.io as scio
import random
from torch.utils import data
from sklearn.model_selection import train_test_split

"""
使用CGAN进行肌肉电信号生成:
生成器
输入:标签和随机噪声
输出:肌肉电信号

判别器
输入:生成的信号和对应的标签; 真实的信号和对应的标签
输出:判别结果(要求输出结果和标签对应)
"""

"""
数据预处理
"""

"""
LoadMat函数:用于读取矩阵(下载的数据集为.mat格式)
提供矩阵的路径，读取该矩阵，并返回emg和label
注意:label来自于restimulus
但是部分数据集中并没有提供restimulus，我们采用stimulus代替
"""


def LoadMat(path):
    mat = scio.loadmat(path)
    emg = mat['emg']
    label = []
    try:
        label = mat['restimulus']
    except:
        label = mat['stimulus']
    return emg, label


"""
cut函数:用于将数据集进行切割
将数据集切割为长度为200，通道数为n的二维数组
注意不同数据集中通道数可能不同
以db2为例，其通道数为12
步长为50
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
count_unique_labels函数:用于统计完切割后的数据集不同标签的数据的数量
根据统计结果，我们会随机删除一部分数据
例如标签0可能会占所有标签的比重过大，我们会予以部分删除
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

def one_hot(x, class_count=41):
    return torch.eye(class_count)[x, :].squeeze()

# 读取矩阵，并存储emg和label
E1_emg, E1_label = LoadMat('D:/Desktop/data/ninapro/db2/DB2_s1/S1_E1_A1.mat')
E2_emg, E2_label = LoadMat('D:/Desktop/data/ninapro/db2/DB2_s1/S1_E2_A1.mat')
E3_emg, E3_label = LoadMat('D:/Desktop/data/ninapro/db2/DB2_s1/S1_E3_A1.mat')


# 将同一志愿者的所有肌电信号数据进行切割后合并
emgs, labels = cut(E1_emg, E1_label)
E2_emg, E2_label = cut(E2_emg, E2_label)
E3_emg, E3_label = cut(E3_emg, E3_label)
emgs.extend(E2_emg)
emgs.extend(E3_emg)
labels.extend(E2_label)
labels.extend(E3_label)

# 查看数据集大小
print('处理前样本数量:'+str(len(emgs)))  # 肌电信号数据集长度为102430

# 发现标签为0的样本数量过多，剔除一部分
# 找到所有标签为 0 的样本的索引
label_0_indices = [i for i, label in enumerate(labels) if label == 0]

# 随机选择 100 个标签为 0 的样本的索引
selected_indices = random.sample(label_0_indices, min(53600, len(label_0_indices)))

# 保留不在选定索引中的样本和标签
emgs = [emgs[i] for i in range(len(emgs)) if i not in selected_indices]
labels = [labels[i] for i in range(len(labels)) if i not in selected_indices]

count_unique_labels(labels)  # 统计不同标签的数据样本量
print('处理后样本数量:'+str(len(emgs)))

# 切割训练集和测试集(6:4)
train_emgs, test_emgs, train_labels, test_labels = train_test_split(emgs, labels, test_size=0.4, random_state=42)

print('训练集大小为:'+str(len(train_labels)))
print('测试集大小为:'+str(len(test_labels)))

# 将数据集转换为tensor
train_emgs = np.array(train_emgs)
train_labels = np.array(train_labels)
test_emgs = np.array(test_emgs)
test_labels = np.array(test_labels)
train_emgs = torch.tensor(train_emgs)
train_labels = torch.tensor(train_labels, dtype=torch.int64)
test_emgs = torch.tensor(test_emgs)
test_labels = torch.tensor(test_labels, dtype=torch.int64)

# 查看训练集和测试集的形状和数据类型
print('训练集样本的形状为:'+str(train_emgs.shape))
print('训练集标签的形状为:'+str(test_emgs.shape))
print('测试集样本的形状为:'+str(train_labels.shape))
print('测试集标签的形状为:'+str(test_labels.shape))

print('样本的数据类型为:'+str(train_emgs.dtype))  # emg信号的数据类型为float32
print('标签的数据类型为:'+str(train_labels.dtype))  # label的数据类型为int64


"""
定义数据集
"""
class EMG_dataset(data.Dataset):
    def __init__(self, emgs, labels):
        self.emgs = emgs
        self.labels = labels

    def __getitem__(self, index):
        emg = emgs[index]
        label = labels[index]
        label = one_hot(label)
        return emg, label

    def __len__(self):
        return len(self.emgs)


# 创建训练集的 EMG_dataset 和 DataLoader
train_dataset = EMG_dataset(train_emgs, train_labels)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 创建测试集的 EMG_dataset 和 DataLoader
test_dataset = EMG_dataset(test_emgs, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# 取出一个批次的数据查看其形状
emg, label = next(iter(train_dataloader))
print('一个批次的样本的形状:'+str(emg.shape))  # batch, 200, 12
print('一个批次的标签的形状:'+str(label.shape))  # batch, 1

"""
定义CGAN生成器
"""

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(100, 128 * 50 * 3)
        self.bn1 = nn.BatchNorm1d(128 * 50 * 3)

        self.linear2 = nn.Linear(41, 128 * 50 * 3)
        self.bn2 = nn.BatchNorm1d(128 * 50 * 3)

        self.deconv1 = nn.ConvTranspose2d(256, 128,
                                          kernel_size=(3, 3),
                                          stride=1,
                                          padding=1)  # 得到的数据形状为128*50*3
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1)  # 得到的图像为64*100*6
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1)  # 得到的图像为(1, 200, 12)

    def forward(self, x, y):  # x为随机噪声输入(长度为100)，y为标签输入(长度41, one-hot)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = x.view(-1, 128, 50, 3)  # 形状为(batch, 128, 50, 3)

        y = F.relu(self.linear2(y))
        y = y.squeeze(1)
        y = self.bn2(y)
        y = y.view(-1, 128, 50, 3)  # 形状为(batch, 128, 50, 3)

        x = torch.cat([x, y], axis=1)  # batch, 256, 50, 3

        x = x.view(-1, 256, 50, 3)
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.deconv2(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = self.deconv3(x)
        x = torch.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(41, 1 * 200 * 12)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128*49*2, 1)

    def forward(self, y, x):  # y为条件, x代表肌电信号
        y = self.linear(y)
        y = F.leaky_relu(y)
        y = y.view(-1, 1, 200, 12)
        x = torch.cat([y, x], axis=1)  # batch, 2, 200, 12
        x = F.dropout2d(F.leaky_relu(self.conv1(x)), p=0.3)
        x = F.dropout2d(F.leaky_relu(self.conv2(x)), p=0.3)
        x = self.bn(x)
        x = x.view(-1, 128 * 49 * 2)  # (batch, 128, 200, 12) -> (batch, 128*200*12)
        x = torch.sigmoid(self.fc(x))
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'

gen = Generator().to(device)
dis = Discriminator().to(device)

loss_fn = torch.nn.BCELoss()
d_optimizer = torch.optim.Adam(dis.parameters(), lr=0.0001)
g_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0001)


def generate_and_save_images(model, epoch, label_input, noise_input):
    pred = model(noise_input, label_input).cpu().detach().numpy()
    fig = plt.figure(figsize=(10, 10))  # 调整画布大小

    for i in range(pred.shape[0]):
        plt.subplot(10, 10, i + 1)  # 调整子图数量和排列方式
        plt.imshow(pred[i].reshape(200, 12), cmap='gray')  # 重新形状为 200x12 并绘制
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()  # 不显示图像，直接关闭画布


# 随机生成用于模型输入的噪声和标签
noise_seed = torch.randn(16, 100, device=device)
label_seed = torch.randint(0, 41, size= (16,))
label_seed_onehot = one_hot(label_seed).to(device)
print('随机生成的16个标签为:', label_seed)


D_loss = []
G_loss = []

for epoch in range(100):
    D_epoch_loss = 0
    G_epoch_loss = 0
    correct = 0
    sum = 0
    count = len(train_dataloader.dataset)
    for step, (img, label) in enumerate(train_dataloader):
        img = img.to(device)  # (batch, 200, 12)
        label = label.to(device)  # (batch, 41)

        size = img.shape[0]
        random_seed = torch.randn(size, 100, device=device)

        d_optimizer.zero_grad()
        img_reshaped = img.unsqueeze(1)
        real_output = dis(label, img_reshaped)  # (batch, 41) (batch, 200, 12)

        d_real_loss = loss_fn(real_output, torch.ones_like(real_output, device=device))

        d_real_loss.backward()

        generated_img = gen(random_seed, label)
        fake_output = dis(label, generated_img.detach())

        d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output, device=device))
        d_fake_loss.backward()

        disc_loss = d_real_loss + d_fake_loss
        d_optimizer.step()

        g_optimizer.zero_grad()
        fake_output = dis(label, generated_img)
        gen_loss = loss_fn(fake_output, torch.ones_like(fake_output, device=device))
        gen_loss.backward()
        g_optimizer.step()

        with torch.no_grad():
            D_epoch_loss += disc_loss.item()
            G_epoch_loss += gen_loss.item()

            # 判别器准确率dis_acc
            real_output[real_output >= 0.5] = 1
            real_output[real_output < 0.5] = 0
            correct += real_output.eq(1).sum().item()
            fake_output[fake_output >= 0.5] = 1
            fake_output[fake_output < 0.5] = 0
            correct += fake_output.eq(0).sum().item()
            sum += real_output.size(0)
            sum += fake_output.size(0)
    acc = correct / sum
    print("acc:"+str(acc))

    with torch.no_grad():
        D_epoch_loss /= count
        G_epoch_loss /= count
        D_loss.append(D_epoch_loss)
        G_loss.append(G_epoch_loss)

    print('Epoch: ', epoch)

plt.plot(D_loss, label='D_loss')
plt.plot(G_loss, label='G_loss')
plt.show()


