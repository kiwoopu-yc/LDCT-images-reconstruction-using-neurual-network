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
from torch.utils.data import Dataset, DataLoader, ConcatDataset


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
            nn.Conv2d(1, 16, kernel_size=3, stride=3, padding=1, dilation=1),      #b,16,254,254
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),         # (b, 16, 127, 127)

            nn.Conv2d(16, 8,  kernel_size=3, stride=2, padding=1, dilation=1),  # (b, 8, 63, 63)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 62, 62)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, dilation=1),  # (b, 16, 17, 17)   #out = (in-1)*str-2p+k+p
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=6, stride=2, padding=1, dilation=1),  # (b, 1, 28, 28)
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc21 = nn.Linear(256, 128)    # 均值mean
        self.fc22 = nn.Linear(256, 128)    # 方差var
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
        # if torch.cuda.is_available():
        #     eps = torch.cuda.FloatTensor(std.size()).normal_()    # 从标准的正态分布中随机采样一个eps
        # else:
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return self.decoder(z), mu, logvar


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
        a = pickle.load(open(self.sinofbpfiles[index], 'rb'))
        ori = a[0]
        target = a[1]
        # ori = ori.reshape((1, 262144))
        # target = target.reshape((1, 262144))
        ori = ori[np.newaxis, :]
        target = target[np.newaxis, :]
        return ori, target

    def __len__(self):
        # 返回投影的数量
        return len(self.sinofbpfiles)


learning_rate = 0.00002
num_epoches = 10000
weight_decay = 1e-5
batch_size = 2
epoch = 0
if __name__ == '__main__':

    data_path = 'C:/Users/pyc/Desktop/72_290_1160/4test_ham72_ref/'

    train_datasets = sino2fbpDataset(root=data_path)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

    #autoencoder = VAE().cuda()
    autoencoder = auto_encoder()
    #autoencoder = torch.load('C:/Users/pyc/Desktop/model/vae_72_2.pkl')
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_func = nn.MSELoss(reduction='sum')    #reduction = 'mean'    reduction = 'sum'
    #noise = torch.rand(batch_size, 1, 512, 512)
    for i in range(0, 500):
        for ori, target in train_dataloader:

            # ori = data[0]
            # target = data[1]
            #ori, target = data

            ori = ori.float()
            target = target.float()
            #print(ori.size())
            #ori = torch.mul(ori + 0.25, 0.1 * noise)    #添加噪声
            ori = Variable(ori)
            target = Variable(target)
            #out, mu, logvar  = autoencoder(ori)
            out = autoencoder(ori)
            loss = loss_func(out, target)
            # kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())    #KL
            # loss = loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch += 1

            if epoch % 100 == 0:
                print('epoch{} loss is {:.4f}'.format(epoch, loss.item()))

    # temp = out.detach().numpy()
    # plt.imshow(temp, cmap='gray')
    # plt.show()
    torch.save(autoencoder, 'C:/Users/pyc/Desktop/model/ae_72_1.pkl')
