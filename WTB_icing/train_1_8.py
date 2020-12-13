# -*- coding: utf-8 -*-
# @Time    : 3/22/20 19:15
# @Author  : huangting
# @FileName: train.py
# @Software: PyCharm

import argparse
import os
import dataLoader
from model import CNN_LSTM
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from keras.utils.np_utils import to_categorical

parser = argparse.ArgumentParser(description='风机结冰预测')
parser.add_argument("--model", default="CNN-LSTM", type=str, help="需要训练的模型")
parser.add_argument("--bs", default=100, type=int, help="batchsize")
parser.add_argument("--lr", default=0.0001, type=float, help="学习率")
parser.add_argument("--ts", default=1, type=int, help="网络处理数据的时间步长")
parser.add_argument("--dl", default=0, type=int, help="延时预测的长度")
parser.add_argument("--lb", default=1, type=int, help="滑窗处理的步长")
parser.add_argument("--dr", default="./data", type=str, help="数据所在的根路径")
parser.add_argument("--rr", default="./data/r/r_0.08.txt", type=str, help="注意力矩阵文件路径")
parser.add_argument("--ep", default=100, type=int, help="训练的总轮数")
parser.add_argument('--re', default=False, help='是否从checkpoint处开始训练')
parser.add_argument("--ckpt", default="./checkpoints/res1_52/CNN-LSTM-82.t7", help="checkpoint的地址")
parser.add_argument("--gpu", default=False, help="是否使用GPU进行训T练")
parser.add_argument("--cuda", default="0，1，2，3", type=str, help="用于训练GPU的代号")

cfg, unknown = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda # 配置环境中的GPUF

start_epoch = 0 # 开始训练的轮数
best_test_acc = 0 # 最好的测试准确率
best_test_acc_epoch = 0 # 最好的测试准确率时的轮数

print("--------------加载训练数据中------------")
trainRoot = cfg.dr + "/train"
X_train, y_train, mask_train = dataLoader.window_process(trainRoot, cfg.rr, cfg.dl, cfg.ts, cfg.lb)
X_train = np.array(X_train)
y_train = np.array(y_train)
mask_train = np.array(mask_train)
y_train = to_categorical(y_train, 2)

# 对数据进行随机打乱
permutation = np.random.permutation(X_train.shape[0])
X_train = X_train[permutation, :, :, :]
y_train = y_train[permutation, :]
mask_train = mask_train[permutation, :, :]

trainData_sum = len(X_train) # 训练样本总数
print("--------------加载训练数据完成------------")

print("--------------加载测试数据中------------")
testRoot = cfg.dr + "/test"
X_test, y_test, mask_test = dataLoader.window_process(testRoot, cfg.rr, cfg.dl, cfg.ts, cfg.lb)
X_test = np.array(X_test)
y_test = np.array(y_test)
mask_test = np.array(mask_test)
y_test = to_categorical(y_test, 2)

# 对数据进行随机打乱
permutation = np.random.permutation(X_test.shape[0])
X_test = X_test[permutation, :, :, :]
y_test = y_test[permutation, :]
mask_test = mask_test[permutation, :, :]

testData_sum = len(X_test) # 训练样本总数
print("--------------加载测试数据完成------------")

use_cuda = torch.cuda.is_available()  # 环境中是否有GPU
net = CNN_LSTM.CNN_LSTM(cfg.ts)

if use_cuda and cfg.gpu:
    net = net.cuda()
    net = nn.DataParallel(net)

if cfg.re:
    print('------------------------------')
    print('==> 加载checkpoint ')
    if not os.path.exists(cfg.ckpt):
        raise AssertionError['找不到路径']
    checkpoint = torch.load(cfg.ckpt)
    net.load_state_dict(checkpoint['net'])
    best_test_acc = checkpoint['best_test_acc']
    print('best_test_acc is %.4f%%'%best_test_acc)
    best_test_acc_epoch = checkpoint['best_test_acc_epoch']
    print('best_test_acc_epoch is %d'%best_test_acc_epoch)
    start_epoch = checkpoint['best_test_acc_epoch'] + 1
else:
    print('------------------------------')
    print('==> 构建新的模型')

optimizer = optim.Adam(net.parameters(), lr=cfg.lr)  # 实例化梯度下降算法
MSELoss = nn.MSELoss()  # 实例化loss

def train(epoch):
    global train_acc
    trainloss = 0.0
    total = 0
    correct = 0

    for i in range(0, trainData_sum, cfg.bs):
        if trainData_sum - i >= cfg.bs:
            inputs = X_train[i:(i+cfg.bs), :, :, :]
            target = y_train[i:(i+cfg.bs), :]
            mask = mask_train[i:(i+cfg.bs), :, :]
        else:
            inputs = X_train[i:trainData_sum, :, :, :]
            target = y_train[i:trainData_sum, :]
            mask = mask_train[i:trainData_sum, :, :]

        inputs = torch.Tensor(inputs)
        target = torch.Tensor(target)
        mask = torch.Tensor(mask)

        if use_cuda and cfg.gpu:
            inputs = inputs.cuda()
            target = target.cuda()
            mask = mask.cuda()

        outputs = net(inputs, mask)
        optimizer.zero_grad()
        loss = MSELoss(target, outputs[8])
        loss.backward()
        optimizer.step()
        trainloss += loss.item()

        _, predicted = torch.max(outputs[8].data, 1)
        _, trueValue = torch.max(target.data, 1)
        total += target.size(0)
        correct += predicted.eq(trueValue.data).sum()

       # print("第" + str(epoch) + "轮，第" + str(i) + "个batch已经训练完成")

    train_acc = 100.0 * int(correct.data) / total
    print("训练误差为 %.3f" % trainloss)
    print('在 %d 个样本中, %d 个被准确预测' % (total, correct))
    print("训练准确率为 %.4f%%" % train_acc)
    print("第" + str(epoch) + "训练已经完成")

    return

def test(epoch):
    global test_acc
    global best_test_acc
    global best_test_acc_epoch
    net.eval()
    total = 0
    correct = 0
    confuse = torch.zeros(2, 2)  # 定义混淆矩阵

    for i in range(0, testData_sum, cfg.bs):
        if testData_sum - i >= cfg.bs:
            inputs = X_test[i:(i+cfg.bs), :, :, :]
            target = y_test[i:(i+cfg.bs), :]
            mask = mask_test[i:(i + cfg.bs), :, :]
        else:
            inputs = X_test[i:testData_sum, :, :, :]
            target = y_test[i:testData_sum, :]
            mask = mask_test[i:testData_sum, :, :]


        inputs = torch.Tensor(inputs)
        target = torch.Tensor(target)
        mask = torch.Tensor(mask)

        if use_cuda and cfg.gpu:
            inputs = inputs.cuda()
            target = target.cuda()

        with torch.no_grad():
            outputs = net(inputs, mask)

        _, predicted = torch.max(outputs[8].data, 1)
        _, trueValue = torch.max(target.data, 1)

        for j in range(predicted.size()[0]):
            confuse[predicted[j], trueValue[j]] += 1

        total += target.size(0)
        correct += predicted.eq(trueValue.data).sum()

    for categorical in range(2):
        confuse[categorical] = confuse[categorical] / confuse[categorical].sum()
    print(confuse.data)
    test_acc = 100.0 * int(correct.data) / total
    print('在 %d 个样本中, %d 个被准确预测' % (total, correct))
    print('测试准确率为 %.4f%%' % test_acc)
    print("一轮测试已经完成")

    if test_acc > best_test_acc:
        print('保存新的checkpoint')
        print("best_test_acc: %0.4f%%" % test_acc)
        print('best_test_epoch: %d ' % epoch)
        state = {
                'net': net.state_dict(),
                'best_test_acc': test_acc,
                'best_test_acc_epoch': epoch,
        }
        torch.save(state, os.path.join('./checkpoints/res_1_8', cfg.model + '-' + str(epoch) + '.t7'))
        best_test_acc = test_acc
        best_test_acc_epoch = epoch
        print('更新完成')

    return

if __name__ == "__main__":

    print('--------------------------------------------------------')
    print('-------------batch size: %d' % cfg.bs)
    print('-------------backbone: %s' % cfg.model)
    print('-------------total epoch: %d' % cfg.ep)
    print('-------------device: %s' % ('GPU' if cfg.gpu and use_cuda else 'CPU'))
    print('--------------------------------------------------------')
    for epoch in range(start_epoch, cfg.ep):
        train(epoch)
        test(epoch)




