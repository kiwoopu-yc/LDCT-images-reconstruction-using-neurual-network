import numpy as np
import os
import torch
import pickle
import mat4py
import scipy.io as scio
#a = np.load('C:/Users/59186/Desktop/output_npy/pic1-1-views.npy', allow_pickle=True, encoding="latin1")
input_path = 'C:/Users/59186/Desktop/sino2ori/train_sino_npy/'
target_path = 'C:/Users/59186/Desktop/sino2ori/train_fbp_npy/'
output_path = 'C:/Users/59186/Desktop/sino2ori/sino_connect_FBP/'
file1 = os.listdir(input_path)
file2 = os.listdir(target_path)
for file in file1:
    first_name, second_name = os.path.splitext(file)
    # 拆分.mat文件的前后缀名字，注意是**路径**
    a = first_name.split('_', 1)
    b = a[0]
    position = input_path + '\\' + file  # 构造绝对路径，"\\"，其中一个'\'为转义符
    position1 = target_path + '\\' + b + second_name
    print(position)
    a = np.load(position)
    b = np.load(position1)
    print(a.shape)
    #b = np.transposea(b, 0, 1)
    # a = a.reshape((1, 52992))
    # b = b.reshape((1, 262144))
    c = ((a.astype(np.float), b.astype(np.float)))

    output = open(output_path + first_name + '.pkl', 'wb')
    pickle.dump(c, output)
    print(c[0])


# matrix = np.load('yourfile.npy')
# print(matrix)