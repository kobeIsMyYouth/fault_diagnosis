# -*- coding: utf-8 -*-
# @Time    : 3/28/20 15:30
# @Author  : huangting
# @FileName: dataExtract.py
# @Software: PyCharm

import numpy as np
from openpyxl import Workbook
import pandas as pd


def stand_train(data):
    """
    对训练集的标准化操作
    :param data: 需要标准化的数据
    :return: 标准化之后的数据
    """
    mean = np.mean(data, 0)
    var = np.std(data, 0)
    print(data[:, 0:(data.shape[1] - 1)].shape)
    data[:, 0:data.shape[1] - 1] = (data[:, 0:data.shape[1] - 1] - mean[0:data.shape[1] - 1]) / var[0:
            data.shape[1] - 1]

    return mean, var, data

def stan_test(mean, var, data):
    """
    对测试集的标准化操作
    :param mean: 训练集的均值
    :param var: 测试集的均值
    :param data: 需要标准化的数据
    :return: 标准化之后的数据
    """
    data[:, 0:data.shape[1] - 1] = (data[:, 0:data.shape[1] - 1] - mean[0:data.shape[1] - 1]) / var[0:
            data.shape[1] - 1]
    return data


if __name__=="__main__":

    data = pd.read_csv("./data/test/21_stan.csv", header = None)

    print(data)
    data = data.values

    flags = [0]

    for i in range(data.shape[0]-1):
        if data[i,26] == 1 and data[i+1,26] == 0:
            flags.append(i+1)
    flags.append(data.shape[0])

    print(flags)

    for j in range(len(flags)-1):
        np.savetxt("data/test/21_" + str(j) + ".csv", data[flags[j]:flags[j+1],:], delimiter = ',')

    # data_train = pd.read_csv("./data/train/15.csv")
    #
    # data_train = np.array(data_train.values)
    #
    # mean, var, data_train = stand_train(data_train)
    #
    # data_test = pd.read_csv("./data/test/21.csv")
    #
    # data_test = np.array(data_test.values)
    #
    # data_test = stan_test(mean, var, data_test)
    #
    # np.savetxt("data/train/15_stan.csv", data_train, delimiter=',')
    #
    # np.savetxt("data/test/21_stan.csv", data_test, delimiter=',')





