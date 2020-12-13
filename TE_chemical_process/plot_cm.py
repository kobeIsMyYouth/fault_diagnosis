# -*- coding: utf-8 -*-
# @Time    : 4/5/20 4:27 PM
# @Author  : huangting
# @FileName: plot_cm.py
# @Software: PyCharm

import itertools
import matplotlib.pyplot as plt
import numpy as np
def plot_confusion_matrix(cm, classes, saveRoot, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if 0 < cm[i, j] < 1:
            fmt = '.4f' if normalize else 'd'
        else:
            fmt = ".0f" if normalize else "d"
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True categories')
    plt.xlabel('Predicted categories')
    plt.savefig(saveRoot)