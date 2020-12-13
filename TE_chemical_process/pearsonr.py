# -*- coding: utf-8 -*-
# @Time    : 3/29/20 22:58
# @Author  : huangting
# @FileName: pearsonr.py
# @Software: PyCharm

import readDataUtils as rdu
from scipy.stats import pearsonr
import numpy as np
import os

def r(path, y):
    """
    计算一个故障数据的相关性系数
    :param path: 数据所在的路径
    :return: 每一个属性是否与标签相关 r[0,1,0,0,1,...,0,0,0]
    """
    X = rdu.readExcel(path)


    r = [0 for i in range(52)]

    for i in range(52):

        a = abs(pearsonr(X[:, i], y)[0])

        if a >= 0.11:
            r[i] = 1

    return r

def rs(root):
    """
    计算所有数据的相关系数矩阵
    :param root: 数据所在的根路径
    :return: R(22 X 52) [
        [0,1,0,1,0,1...0,0],
        [],
        [],
    ]
    """
    R = np.zeros((22,52))
    files = os.listdir(root)

    for file in files:
        if not file.startswith("."):

            id = int(file[1:3])

            if id == 0:
                y = [1 for i in range(500)]
                print(y)
            else:
                y = [0 if i < 20 else 1 for i in range(500)]

            filePath = root + "/" + file
            pr = r(filePath, y)
            R[id] = pr

    return R

if __name__=="__main__":
    root = "./data/train"
    R = rs(root)
    sum = np.sum(R, 1)
    print(R)
    print(R.shape)
    print(sum)

    np.savetxt("./data/r/r_0.11.txt", R)