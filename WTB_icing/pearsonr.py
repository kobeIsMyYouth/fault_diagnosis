# -*- coding: utf-8 -*-
# @Time    : 3/29/20 22:58
# @Author  : huangting
# @FileName: pearsonr.py
# @Software: PyCharm

from scipy.stats import pearsonr
import numpy as np
import os
import pandas as pd

def r(path):
    """
    计算一个故障数据的相关性系数
    :param path: 数据所在的路径
    :return: 每一个属性是否与标签相关 r[0,1,0,0,1,...,0,0,0]
    """
    data = pd.read_csv(path, header = None).values
    X = data[:, 0:-1]
    Y = data[:, -1]
    r = [0 for i in range(26)]

    for i in range(26):

        a = abs(pearsonr(X[:, i], Y)[0])

        if a >= 0.17:
            r[i] = 1

    return r


if __name__=="__main__":

    r = r("./data/15_stan.csv")

    R = np.zeros((2,26))
    R[1] = r
    print(R)

    np.savetxt("./data/r/r_0.17.txt", R)