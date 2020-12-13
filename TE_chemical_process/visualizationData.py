# -*- coding: utf-8 -*-
# @Time    : 3/22/20 10:11
# @Author  : huangting
# @FileName: visualizationData.py
# @Software: PyCharm

import readDataUtils as rdu
import matplotlib.pyplot as plt
import numpy as np

data = rdu.readExcel("./data/test/d05_te.xlsx")
d = rdu.readExcel("./data/test/d00_te.xlsx")

x = np.linspace(0, 960, 960)

for i in range(52):
    plt.figure()
    plt.plot(x, data[:, i])
    plt.axvline(x=160, ls="-", c="r")
    plt.plot(x, d[:, i], c="b")

    plt.savefig("./images/dataVisualization_05_" + str(i) + ".jpg")

    plt.show()

