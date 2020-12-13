# -*- coding: utf-8 -*-
# @Time    : 3/28/20 15:30
# @Author  : huangting
# @FileName: dataExtract.py
# @Software: PyCharm

import numpy as np
from openpyxl import Workbook

def readDat(path):
    """
    读取。dat文件
    :param path: 文件所在路径
    :return: numpy数组
    """
    data = np.loadtxt(path)

    return data

def stand_train(data):
    """
    对训练集的标准化操作
    :param data: 需要标准化的数据
    :return: 标准化之后的数据
    """
    mean = np.mean(data, 0)
    var = np.std(data, 0)
    data = (data - mean) / var

    return mean, var, data

def stan_test(mean, var, data):
    """
    对测试集的标准化操作
    :param mean: 训练集的均值
    :param var: 测试集的均值
    :param data: 需要标准化的数据
    :return: 标准化之后的数据
    """
    data = (data - mean) / var
    return data

def writeDataAsExcel(path, data, fileName):
    """
    将numpy数组保存为表格
    :param path: 保存的文件夹
    :param data: 需要保存的数据
    :param fileName: 保存的文件名
    :return:
    """
    assert data.ndim == 2, \
        "数组的维度必须为2维"
    wb = Workbook()
    ws = wb.active
    for row in range(data.shape[0]): # 行
        for col in range(data.shape[1]): # 列
            ws.cell(row=row + 1, column=col + 1).value = data[row][col]
    wb.save(path + "/" + fileName)
    return

if __name__=="__main__":

    rootPath = "/Users/huangting/Documents/github/tennessee-eastman-profBraatz"

    data0 = readDat(rootPath + "/train/d00.txt").T
    dataadd = data0[0:20, :]


    for i in range(22):
        if i < 10:
            data_train_path = rootPath + "/train/d0" + str(i) + ".txt"
            data_test_path = rootPath + "/test/d0" + str(i) + "_te.txt"
        else:
            data_train_path = rootPath + "/train/d" + str(i) + ".txt"
            data_test_path = rootPath + "/test/d" + str(i) + "_te.txt"

        data_train = readDat(data_train_path)
        data_test = readDat(data_test_path)

        if i == 0:
            data_train = data_train.T

        if i != 0:
            data_train = np.vstack((dataadd, data_train))

        writeDataAsExcel("./data/orig/train/", data_train, "d" + str(i) + ".xlsx")
        writeDataAsExcel("./data/orig/test/", data_test, "d" + str(i) + "_te.xlsx")

        mean, var, data_train_stan = stand_train(data_train)
        data_test_stan = stan_test(mean, var, data_test)

        writeDataAsExcel("./data/train/", data_train_stan, "d" + str(i) + ".xlsx")
        writeDataAsExcel("./data/test/", data_test_stan, "d" + str(i) + "_te.xlsx")

        print(i)




