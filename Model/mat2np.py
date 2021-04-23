import pickle

import numpy as np
import os
import mat4py
import scipy.io as scio
#a = np.load('C:/Users/59186/Desktop/output_npy/pic1-1-views.npy', allow_pickle=True, encoding="latin1")
origin_path = 'C:/Users/pyc/Desktop/72_290_1160/mat/other_fbp_72/'
origin_path1 = 'C:/Users/pyc/Desktop/72_290_1160/mat/other_fbp_reference/'

output_path = 'C:/Users/pyc/Desktop/72_290_1160/other_fbp_ref/'
files = os.listdir(origin_path)
for file in files:
    first_name, second_name = os.path.splitext(file)
    fa = first_name.split('_', 1)
    fb = fa[0]
    # 拆分.mat文件的前后缀名字，注意是**路径**
    position = origin_path + '\\' + file  # 构造绝对路径，"\\"，其中一个'\'为转义符
    position1 = origin_path1 + '\\' + fb + '_reference.mat'  # 构造绝对路径，"\\"，其中一个'\'为转义符
    print(position)
    a = mat4py.loadmat(position)
    b = mat4py.loadmat(position1)
   # print(a)
    mat_t = a['rec']
    mat_t = np.reshape(mat_t, (512, 512))
    mat_t1 = b['P']
    mat_t1 = np.reshape(mat_t1, (512, 512))
    temp = (mat_t.astype(np.float), mat_t1.astype(np.float))
    output = open(output_path + first_name + '.pkl', 'wb')
    pickle.dump(temp, output)
# for file in files:
#     first_name, second_name = os.path.splitext(file)
#     fa = first_name.split('_', 1)
#     fb = fa[0]
#     position = origin_path + '\\' + file  # 构造绝对路径，"\\"，其中一个'\'为转义符
#     print(position)
#     a = mat4py.loadmat(position)
#     mat_t = a['rec']
#     np.save(output_path + first_name + '.npy', mat_t)

    #     mat_t = np.reshape(mat_t, (512, 512))
    #     a = mat4py.loadmat(position)
# matrix = np.load('yourfile.npy')
# print(matrix)
