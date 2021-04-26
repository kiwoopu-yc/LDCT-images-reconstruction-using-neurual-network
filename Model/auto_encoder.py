import os
import pickle

import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

def normalizeto255(operator):
    min = np.min(operator)
    max = np.max(operator)
    range = max - min
    operator = operator - min
    operator = operator/range*255
    return operator

class auto_encoder(nn.Module):
    def __init__(self):
        super(auto_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc21 = nn.Linear(256, 128)    # 均值
        self.fc22 = nn.Linear(256, 128)    # 方差
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def decoder(self, z):
        h3 = F.relu(self.fc3(z))
        x = F.tanh(self.fc4(h3))
        return x
    # 重新参数化
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()   # 计算标准差
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()    # 从标准的正态分布中随机采样一个eps
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return self.decoder(z)


class sino2fbpDataset(Dataset):
    """
     root：存放地址根路径
    """
    def __init__(self, root):
        # 这个list存放所有图像的地址
        self.sinofbpfiles = np.array([x.path for x in os.scandir(root) if
                                      x.name.endswith(".pkl")])

    def __getitem__(self, index):
        """
                Args:
                    index (int): Index

                Returns:
                    tuple: (image, target) where target is index of the target class.
                """
        # 读取图像数据并返回
        # 这里的open_image是读取图像函数，可以用PIL、opencv等库进行读取
        return pickle.load(open(self.sinofbpfiles[index], 'rb'))

    def __len__(self):
        # 返回投影的数量
        return len(self.sinofbpfiles)


learning_rate = 0.00002
num_epoches = 10000
weight_decay = 1e-5
batch_size = 2
epoch = 0
if __name__ == '__main__':

    data_path = 'C:/Users/59186/Desktop/DATA/numpy/fbp_ref/'
    train_datasets = sino2fbpDataset(root=data_path)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

    #autoencoder = VAE()
    autoencoder = torch.load('C:/Users/59186/Desktop/npydataset/model/model_autoencoder2.pkl')
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss(reduction='sum')    #reduction = 'mean'    reduction = 'sum'
    noise = torch.rand(batch_size, 1, 512, 512)
    for i in range(0, 5000):
        for data in train_dataloader:

            ori = data[0]
            target = data[1]
            ori = ori.float()
            target = target.float()
            #ori = torch.mul(ori + 0.25, 0.1 * noise)    #添加噪声
            ori = Variable(ori)
            target = Variable(target)
            out = autoencoder(ori)
            loss = loss_func(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch += 1

            if epoch % 1000 == 0:
                print('epoch{} loss is {:.4f}'.format(epoch, loss.item()))

    # temp = out.detach().numpy()
    # plt.imshow(temp, cmap='gray')
    # plt.show()
    torch.save(autoencoder, 'C:/Users/59186/Desktop/npydataset/model/model_autoencoder2.pkl')
