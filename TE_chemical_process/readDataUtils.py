# -*- coding: utf-8 -*-
# @Time    : 3/22/20 10:08
# @Author  : huangting
# @FileName: readDataUtils.py
# @Software: PyCharm

import openpyxl
import numpy as np

def readExcel(wbname, sheetname=""):
    """
    读取一个excel文件特定表名的数据
    :param wbname: 文件名
    :param sheetname: 表名
    :return：numpy数组
    """
    wb = openpyxl.load_workbook(filename=wbname, read_only=True)
    if (sheetname == ""):
        ws = wb.active
    else:
        ws = wb[sheetname]
    data = []
    for row in ws.rows:
        list = []
        for cell in row:
            aa = str(cell.value)
            if (aa == ""):
                aa = "0"
            list.append(aa)
        data.append(list)

    data = np.array(data, dtype=float)

    return data