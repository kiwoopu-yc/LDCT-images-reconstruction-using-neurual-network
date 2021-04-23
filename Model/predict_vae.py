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
from skimage.measure import compare_psnr

def normalizeto1(operator):
    min = np.min(operator)
    max = np.max(operator)
    range = max - min
    operator = operator -min
    operator = operator/range
    return operator

def normalizeto004(operator):
    for i in range(0,512):
        for j in range(0, 512):
            if operator[i][j] > 0.4 & operator[i][j] < 0:
                operator[i][j] = 0
    return operator

def normalizeto08(operator):
    for i in range(0,512):
        for j in range(0, 512):
            if operator[i][j] > 0.08 & operator[i][j] < -0.08:
                operator[i][j] = 0
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
        # if torch.cuda.is_available():
        #     eps = torch.cuda.FloatTensor(std.size()).normal_()    # 从标准的正态分布中随机采样一个eps
        # else:
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return self.decoder(z)

learning_rate = 0.0002
num_epoches = 100000

if __name__ == '__main__':

    model = torch.load('C:/Users/pyc/Desktop/model/vae_72_1.pkl')
    #model = torch.load('C:/Users/pyc/Desktop/model/vae1.pkl')
    data_path = 'C:/Users/pyc/Desktop/72_290_1160/hamming1160_ref/pic5_fbp.pkl'
    data_path1 = 'C:/Users/pyc/Desktop/72_290_1160/fbp1160/pic5_fbp.npy'
    #data_path1 = 'C:/Users/59186/Desktop/DATA/numpy/head_ct_reference/pic2_reference.npy'
    ##################测试集#####################################
    data_ori290 = 'C:/Users/pyc/Desktop/72_290_1160/hamming290_ref/pic4_fbp.pkl'
    ref290 = 'C:/Users/pyc/Desktop/72_290_1160/fbp290/pic4_fbp.npy'

    data_ori145 = 'C:/Users/pyc/Desktop/72_290_1160/hamming145_ref/pic3_fbp.pkl'
    ref145 = 'C:/Users/pyc/Desktop/72_290_1160/fbp145/pic3_fbp.npy'

    data_ori72 = 'C:/Users/pyc/Desktop/72_290_1160/hamming72_ref/pic1_fbp.pkl'
    ref72 = 'C:/Users/pyc/Desktop/72_290_1160/fbp72/pic1_fbp.npy'

    net = model  # 加载预先训练好的模型
    torch.no_grad()

    def makepre(oripath):
        a = pickle.load(open(oripath, 'rb'))
        a0 = a[0]
        b = torch.from_numpy(a0)
        ori = b.float()
        out = net(ori)
        temp = out.detach().numpy()
        return temp
    def getref(oripath):
        a = pickle.load(open(oripath, 'rb'))
        a1 = a[1]
        return a1
    def getfbp(oripath):
        a = np.load(oripath)
        return a



###########多图对比
    # #1160
    # ax11 = plt.subplot(451),plt.imshow(getref(data_path), cmap='gray'),plt.ylabel('1160 views'),plt.title('Reference'), plt.xticks([]),plt.yticks([])
    # ax12 = plt.subplot(452),plt.imshow(getfbp(data_path1), cmap='gray'),plt.title('FBP'),plt.axis('off')
    # ax13 = plt.subplot(453),plt.imshow(makepre(data_path), cmap='gray'),plt.title('OUT'),plt.axis('off')
    # ax14 = plt.subplot(454),plt.imshow(getref(data_path) - getfbp(data_path1), cmap='gray'),plt.title('Ref - FBP'),plt.axis('off')
    # ax15 = plt.subplot(455),plt.imshow(getref(data_path) - makepre(data_path), cmap='gray'),plt.title('Ref - OUT'),plt.axis('off')
    #
    # #290
    # ax21 = plt.subplot(456), plt.imshow(getref(data_ori290), cmap='gray'), plt.ylabel('290 views'), plt.xticks([]), plt.yticks([])
    # ax22 = plt.subplot(457), plt.imshow(getfbp(ref290), cmap='gray'), plt.axis('off')
    # ax23 = plt.subplot(458), plt.imshow(makepre(data_ori290), cmap='gray'),  plt.axis('off')
    # ax24 = plt.subplot(459), plt.imshow(getref(data_ori290) - getfbp(ref290), cmap='gray'), plt.axis('off')
    # ax25 = plt.subplot(4,5,10), plt.imshow(getref(data_ori290) - makepre(data_ori290), cmap='gray'), plt.axis('off')
    # #145
    # ax31 = plt.subplot(4,5,11), plt.imshow(getref(data_ori145), cmap='gray'), plt.ylabel('145 views'), plt.xticks([]), plt.yticks([])
    # ax32 = plt.subplot(4,5,12), plt.imshow(getfbp(ref145), cmap='gray'), plt.axis('off')
    # ax33 = plt.subplot(4,5,13), plt.imshow(makepre(data_ori145), cmap='gray'), plt.axis('off')
    # ax34 = plt.subplot(4,5,14), plt.imshow(getref(data_ori145) - getfbp(ref72), cmap='gray'), plt.axis('off')
    # ax35 = plt.subplot(4,5,15), plt.imshow(getref(data_ori72) - makepre(data_ori72), cmap='gray'), plt.axis('off')
    # #72
    # ax41 = plt.subplot(4,5,16), plt.imshow(getref(data_ori72), cmap='gray'), plt.ylabel('72 views'), plt.xticks([]), plt.yticks([])
    # ax42 = plt.subplot(4,5,17), plt.imshow(getfbp(ref72), cmap='gray'), plt.axis('off')
    # ax43 = plt.subplot(4,5,18), plt.imshow(makepre(data_ori72), cmap='gray'),  plt.axis('off')
    # ax44 = plt.subplot(4,5,19), plt.imshow(getref(data_ori72) - getfbp(ref72), cmap='gray'), plt.axis('off')
    # ax45 = plt.subplot(4,5,20), plt.imshow(getref(data_ori72) - makepre(data_ori72), cmap='gray'), plt.axis('off')
    #
    # plt.savefig('C:/Users/pyc/Desktop/1.png')
    #
    # plt.show()
#####两图对比
    # ax11 = plt.subplot(221), plt.imshow(getfbp(ref72), cmap='gray'), plt.title('FBP'), plt.axis('off')
    # cut1 = getfbp(ref72)[:60, 100:420]
    # ax12 = plt.subplot(222), plt.imshow(cut1, cmap='gray'), plt.axis('off')
    # ax21 = plt.subplot(223), plt.imshow(makepre(data_ori72), cmap='gray'), plt.title('OUT'), plt.axis('off')
    # cut2 = makepre(data_ori72)[:60, 100:420]
    # ax22 =plt.subplot(224), plt.imshow(cut2, cmap='gray'), plt.axis('off')
    # plt.savefig('C:/Users/pyc/Desktop/2.png')
    #
    # plt.show()


    #################计算信噪比
    input_path = 'C:/Users/pyc/Desktop/72_290_1160/hamming290_ref/'
    #fbppath = 'C:/Users/pyc/Desktop/72_290_1160/fbp290/'
    file1 = os.listdir(input_path)
    for file in file1:
        first_name, second_name = os.path.splitext(file)

        position = input_path + '\\' + file
        #position1 = fbppath + '\\' + first_name + '.npy'
        a = makepre(position)
        b = getref(position)
        p = compare_psnr(a, b, 255)
        print(p)
