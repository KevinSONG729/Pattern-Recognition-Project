import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import scipy.stats as st
import pandas as pd
import operator
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
import sympy as sp
import 线性分类器

fontstyle = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=10)


class node:
    def __init__(self):
        self.height = 0
        self.weight = 0
        self.shoes = 0


class eig_node:
    def __init__(self):
        self.eig = 0
        self.eig_vector = []


class node1:
    def __init__(self):
        self.height = 0
        self.weight = 0


cmpfun = operator.attrgetter('eig')

# 数据预处理  boytrain  boytest  girltrain  girltest  80%训练  20%测试
f1 = open("D:/专业课/模式识别/forstudent/实验数据/genderdata/boy3.txt").read().split()
f1 = np.matrix(f1).reshape(-1, 3)
boycount = len(f1)
f2 = open("D:/专业课/模式识别/forstudent/实验数据/genderdata/girl3.txt").read().split()
f2 = np.matrix(f2).reshape(-1, 3)
girlcount = len(f2)
boytrain_index = random.sample(range(0, boycount), round(0.7 * boycount))
boytest_index = list(set(range(0, boycount)).difference(set(boytrain_index)))
girltrain_index = random.sample(range(0, girlcount), round(0.7 * girlcount))
girltest_index = list(set(range(0, girlcount)).difference(set(girltrain_index)))
boytrain = []
boytest = []
girltrain = []
girltest = []
for i in [boytrain_index, boytest_index, girltrain_index, girltest_index]:
    for j in i:
        a = node()
        if i == boytrain_index or i == boytest_index:
            a.height = f1[j, 0]
            a.weight = f1[j, 1]
            a.shoes = f1[j, 2]
        else:
            a.height = f2[j, 0]
            a.weight = f2[j, 1]
            a.shoes = f2[j, 2]
        if i == boytrain_index:
            boytrain.append(a)
        elif i == boytest_index:
            boytest.append(a)
        elif i == girltrain_index:
            girltrain.append(a)
        else:
            girltest.append(a)


def PCA(boytrain, girltrain):
    train = boytrain + girltrain
    train_pca = np.array([float(train[0].height), float(train[0].weight), float(train[0].shoes)]).reshape(3, 1)
    boy_pca = np.array([float(boytrain[0].height), float(boytrain[0].weight), float(boytrain[0].shoes)]).reshape(3, 1)
    girl_pca = np.array([float(girltrain[0].height), float(girltrain[0].weight), float(girltrain[0].shoes)]).reshape(3, 1)
    train_pca_index = []
    train.pop(0)
    for i in train:
        x = np.array([float(i.height), float(i.weight), float(i.shoes)]).reshape(-1, 1)
        train_pca = np.hstack((train_pca, x))
    for i in boytrain:
        x = np.array([float(i.height), float(i.weight), float(i.shoes)]).reshape(-1, 1)
        boy_pca = np.hstack((boy_pca, x))
    for i in girltrain:
        x = np.array([float(i.height), float(i.weight), float(i.shoes)]).reshape(-1, 1)
        girl_pca = np.hstack((girl_pca, x))
    print("训练集矩阵：")
    print(train_pca)
    height_mean, weight_mean, shoes_mean = np.mean([float(i.height) for i in train]), np.mean(
        [float(i.weight) for i in train]), np.mean([float(i.shoes) for i in train])
    p = p1 = np.array([height_mean, weight_mean, shoes_mean]).reshape(3, 1)
    for i in range(0, len(train)):
        p1 = np.hstack((p1, p))
    train_pca = train_pca - p1  # 中心化
    train_pca_cov = np.cov(train_pca)  # 求协方差
    # print(train_pca_cov)
    a, b = np.linalg.eig(train_pca_cov)  # 求特征值和特征向量
    # print(a)
    # print(b)
    eig_list = []
    for i in range(0, len(a)):
        f = eig_node()
        f.eig = a[i]
        f.eig_vector = b[i]
        eig_list.append(f)
    eig_list.sort(key=cmpfun, reverse=True)
    # print([i.eig for i in eig_list])
    eig_first2 = [i.eig for i in eig_list[0:2]]
    eig_firstvector2 = np.array([i.eig_vector for i in eig_list[0:2]])
    # print(eig_firstvector2)
    train_pca = np.dot(eig_firstvector2, train_pca)
    boy_pca = np.dot(eig_firstvector2, boy_pca)
    girl_pca = np.dot(eig_firstvector2, girl_pca)
    print("降维后训练集：")
    print(train_pca)
    # print(boy_pca)
    # print(girl_pca)
    boytrain_len, girltrain_len = len(boytrain), len(girltrain)
    boytrain, girltrain = [], []
    for i in range(0, boytrain_len):
        x1 = node1()
        x1.height = boy_pca[0][i]
        x1.weight = boy_pca[1][i]
        boytrain.append(x1)
    for i in range(0, girltrain_len):
        x2 = node1()
        x2.height = girl_pca[0][i]
        x2.weight = girl_pca[1][i]
        girltrain.append(x2)
    return eig_firstvector2, boytrain, girltrain, p1

def test_d2(eig_firstvector2, boytest, girltest, p1):
    boytest_pca = np.array([float(boytest[0].height), float(boytest[0].weight), float(boytest[0].shoes)]).reshape(3, 1)
    girltest_pca = np.array([float(girltest[0].height), float(girltest[0].weight), float(girltest[0].shoes)]).reshape(3, 1)
    for i in boytest:
        x = np.array([float(i.height), float(i.weight), float(i.shoes)]).reshape(-1, 1)
        boytest_pca = np.hstack((boytest_pca, x))
    for i in girltest:
        x = np.array([float(i.height), float(i.weight), float(i.shoes)]).reshape(-1, 1)
        girltest_pca = np.hstack((girltest_pca, x))
    boytest_pca = np.dot(eig_firstvector2, boytest_pca)
    girltest_pca = np.dot(eig_firstvector2, girltest_pca)
    boytest_len, girltest_len = len(boytest), len(girltest)
    boytest, girltest = [], []
    for i in range(0, boytest_len):
        x1 = node1()
        x1.height = boytest_pca[0][i]
        x1.weight = boytest_pca[1][i]
        boytest.append(x1)
    for i in range(0, girltest_len):
        x2 = node1()
        x2.height = girltest_pca[0][i]
        x2.weight = girltest_pca[1][i]
        girltest.append(x2)
    return boytest, girltest

def plot(boytest, girltest):
    plt.scatter([i.height for i in boytest], [i.weight for i in boytest], color="blue")
    plt.scatter([i.height for i in girltest], [i.weight for i in girltest], color="red")


if __name__ == '__main__':
    eig_firstvector2, boytrain_d2, girltrain_d2, p1 = PCA(boytrain, girltrain)
    boytest_d2, girltest_d2 = test_d2(eig_firstvector2, boytest, girltest, p1)
    res, res_bayes, w, m1, m2, w0 = 线性分类器.Fisher(boytrain_d2, girltrain_d2, boytest_d2, girltest_d2)
    correct_rate, correct_rate_bayes = 线性分类器.correct(res, res_bayes, boytest_d2, girltest_d2)
    plot(boytest_d2, girltest_d2)
    线性分类器.roc_plot(res, res_bayes)
