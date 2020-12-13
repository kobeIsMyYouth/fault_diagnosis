# -*- coding: utf-8 -*-
# @Time    : 3/22/20 10:05
# @Author  : huangting
# @FileName: dataLoader.py
# @Software: PyCharm

import os
import numpy as np
import pandas as pd

def window_process(rootPath, rPath, delay, timesteps=26, lamb=1):
    """
    对原始数据进行滑窗处理
    :param rootPath: 原始数据的根路径
    :param rPath: 关联矩阵所在的位置
    :param delay: 设定标签的时延
    :param timesteps: 滑窗的跨时间长度
    :param lamb: 滑窗运行的步长
    :return: 经过滑窗处理之后的数据
    """
    data = []  # 存储每一个类别经过滑窗处理后的数据
    labels = []
    files = sorted(os.listdir(rootPath))

    R = np.loadtxt(rPath, dtype=np.float)
    mask = []

    for file in files:
        if not file.startswith("."):
            filePath = rootPath + "/" + file
            rawData = pd.read_csv(filePath).values
            checkpont = np.sum(rawData[:, 26] == 0)
            label = 1
            r = R[label].reshape((1, 26))
            seqLen = len(rawData)
            for i in range(0, seqLen, lamb):
                if (seqLen - i) >= timesteps and (seqLen - i - timesteps) >= delay:
                    x = rawData[i:(i+timesteps), :]
                    shape = x.shape
                    x = x.reshape(1, shape[0], shape[1])
                    data.append(x.tolist())

                    label_index = i + timesteps - 1  # 标签所在的时序位置
                    if label_index >= checkpont:
                        labels.append(label)
                    else:
                        labels.append(0)
                    if label != 0:
                        p = np.array([0 if j < checkpont else 1 for j in range(i, i + timesteps)]).reshape((timesteps, 1))
                    else:
                        p = np.zeros((timesteps, 1))
                    ma = r * p
                    mask.append(ma)

    return data, labels, mask

if __name__ == "__main__":
    rootPath = "./data/train"
    rPath = "./data/r/r_0.04.txt"
    data, labels, mask = window_process(rootPath, rPath, 0, 26, 1)
    print(labels)
