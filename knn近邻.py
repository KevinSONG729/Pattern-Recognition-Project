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

fontstyle = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=10)


class node:
    def __init__(self):
        self.height = 0
        self.weight = 0
        self.shoes = 0


# 数据预处理  boytrain  boytest  girltrain  girltest  80%训练  20%测试
f1 = open("D:/专业课/模式识别/forstudent/实验数据/genderdata/boy3.txt").read().split()
f1 = np.matrix(f1).reshape(-1, 3)
boycount = len(f1)
f2 = open("D:/专业课/模式识别/forstudent/实验数据/genderdata/girl3.txt").read().split()
f2 = np.matrix(f2).reshape(-1, 3)
girlcount = len(f2)
boytrain_index = random.sample(range(0, boycount), round(0.95 * boycount))
boytest_index = list(set(range(0, boycount)).difference(set(boytrain_index)))
girltrain_index = random.sample(range(0, girlcount), round(0.95 * girlcount))
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


def knn_predict(k, boytest, girltest, boytrain, girltrain):
    res = []
    for i in [boytest, girltest]:
        for j in i:
            distance_list = []
            for p in boytrain:
                distance_list.append(
                    [((float(j.height) - float(p.height)) ** 2 + (float(j.weight) - float(p.weight)) ** 2 + (
                            float(j.shoes) - float(p.shoes)) ** 2) ** (1 / 2), 1])
            for p in girltrain:
                distance_list.append(
                    [((float(j.height) - float(p.height)) ** 2 + (float(j.weight) - float(p.weight)) ** 2 + (
                            float(j.shoes) - float(p.shoes)) ** 2) ** (1 / 2), 0])
            distance_list.sort()
            boy_count = 0
            girl_count = 0
            for p in range(0, k):
                if distance_list[p][1] == 1:
                    boy_count += 1
                else:
                    girl_count += 1
            if i == boytest:
                res.append([float(boy_count - girl_count), 1])
            else:
                res.append([float(boy_count - girl_count), 0])
    return res


def correct(res):
    correct_rate = 0
    for i in res:
        if i[0] > 0 and i[1] == 1 or i[0] < 0 and i[1] == 0:
            correct_rate += 1
    correct_rate /= len(res)
    return correct_rate


def roc_plot():
    res1, res2, res3 = sorted(knn_predict(1, boytest, girltest, boytrain, girltrain), reverse=True), \
                       sorted(knn_predict(3, boytest, girltest, boytrain, girltrain), reverse=True), \
                       sorted(knn_predict(5, boytest, girltest, boytrain, girltrain), reverse=True)
    res4, res5, res6 = sorted(knn_predict_only(1, boytrain, girltrain), reverse=True), \
                       sorted(knn_predict_only(3, boytrain, girltrain), reverse=True), \
                       sorted(knn_predict_only(5, boytrain, girltrain), reverse=True)
    correct_rate1, correct_rate2, correct_rate3, \
    correct_rate4, correct_rate5, correct_rate6 = correct(res1), correct(res2), correct(res3), \
                                                  correct(res4), correct(res5), correct(res6)
    fig = plt.figure()
    ax1, ax2, ax3, ax4, ax5, ax6 = fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233), \
                                   fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)
    ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]
    res_list = [res1, res2, res3, res4, res5, res6]
    label_list = ['k = 1, correct_rate = ' + str(correct_rate1), 'k = 3, correct_rate = ' + str(correct_rate2),
                  'k = 5, correct_rate = ' + str(correct_rate3), 'k = 1, correct_rate = ' + str(correct_rate4),
                  'k = 3, correct_rate = ' + str(correct_rate5), 'k = 5, correct_rate = ' + str(correct_rate6)]
    for i in range(0, 6):
        x, y = [], []
        xx, yy = 0, 0
        x.append(xx)
        y.append(yy)
        for j in res_list[i]:
            if i < 3:
                if j[1] == 1:
                    yy += float(1 / len(boytest))
                else:
                    xx += float(1 / len(girltest))
            else:
                if j[1] == 1:
                    yy += float(1 / len(boytrain))
                else:
                    xx += float(1 / len(girltrain))
            x.append(xx)
            y.append(yy)
        if i < 3:
            ax_list[i].plot(x, y, color='red')
        else:
            ax_list[i].plot(x, y, color='blue')
    for i in range(0, 6):
        ax_list[i].set_xlabel = 'FPR'
        ax_list[i].set_ylabel = 'TPR'
        ax_list[i].set_title(label_list[i])


def knn_predict_only(k, boytrain, girltrain):
    boytrain_len = len(boytrain)
    girltrain_len = len(girltrain)
    correct_rate_only = 0
    res_only = []
    for i in reversed(range(0, boytrain_len - 1)):
        boy_temp = boytrain.copy()
        p = boy_temp.pop(i)
        distance_list = []
        for j in boy_temp:
            distance_list.append(
                [((float(j.height) - float(p.height)) ** 2 + (float(j.weight) - float(p.weight)) ** 2 + (
                        float(j.shoes) - float(p.shoes)) ** 2) ** (1 / 2), 1])
        for j in girltrain:
            distance_list.append(
                [((float(j.height) - float(p.height)) ** 2 + (float(j.weight) - float(p.weight)) ** 2 + (
                        float(j.shoes) - float(p.shoes)) ** 2) ** (1 / 2), 0])
        distance_list.sort()
        boy_count = 0
        girl_count = 0
        for j in range(0, k):
            if distance_list[j][1] == 1:
                boy_count += 1
            else:
                girl_count += 1
        res_only.append([float(boy_count - girl_count), 1])
    for i in reversed(range(0, girltrain_len - 1)):
        girl_temp = girltrain.copy()
        p = girl_temp.pop(i)
        distance_list = []
        for j in boytrain:
            distance_list.append(
                [((float(j.height) - float(p.height)) ** 2 + (float(j.weight) - float(p.weight)) ** 2 + (
                        float(j.shoes) - float(p.shoes)) ** 2) ** (1 / 2), 1])
        for j in girl_temp:
            distance_list.append(
                [((float(j.height) - float(p.height)) ** 2 + (float(j.weight) - float(p.weight)) ** 2 + (
                        float(j.shoes) - float(p.shoes)) ** 2) ** (1 / 2), 0])
        distance_list.sort()
        boy_count = 0
        girl_count = 0
        for j in range(0, k):
            if distance_list[j][1] == 1:
                boy_count += 1
            else:
                girl_count += 1
        res_only.append([float(boy_count - girl_count), 0])
    return res_only


def distance(x, y):
    return ((float(x.height) - float(y[0])) ** 2 + (float(x.weight) - float(y[1])) ** 2 + (
            float(x.shoes) - float(y[2])) ** 2) ** (1 / 2)


def scatter_contour():  # height:140-180 weight:30-80 shoes:20-40
    res_x, res_y, res_z = [], [], []
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    for i in range(140, 180):
        for j in range(30, 80):
            for k in range(20, 40):
                x = node()
                x.height, x.weight, x.shoes = i, j, k
                res = knn_predict(5, [x], [], boytrain, girltrain)
                # print((res[0])[0])
                if res[0][0] > 0:
                    ax1.scatter(i, j, k, color='red')
                else:
                    ax1.scatter(i, j, k, color='blue')


def knn_cut(boytrain, girltrain):
    fig = plt.figure()
    ax1, ax2, ax3, ax4 = fig.add_subplot(221, projection='3d'), fig.add_subplot(222, projection='3d'), fig.add_subplot(
        223, projection='3d'), fig.add_subplot(224, projection='3d')
    ax_list = [ax1, ax2, ax3, ax4]
    train = []
    for i in boytrain:
        train.append([i, 1])
    for j in girltrain:
        train.append([j, 0])
    random.shuffle(train)
    train_len = len(train)
    print(train_len)
    train1, train2, train3 = train[0:round(train_len / 3)], train[round(train_len / 3):round(train_len * 2 / 3)], train[
                                                                                                                  round(
                                                                                                                      train_len * 2 / 3):train_len + 1]
    p = 1
    q = 0
    while (p):
        p = 0
        ax_list[q].set_title('sample_num = ' + str(len(train1) + len(train2) + len(train3)))
        for i in train1:
            min = 10000000
            min_label = train2[0][1]
            for j in train2 + train3:
                x = distance(i[0], [j[0].height, j[0].weight, j[0].shoes])
                if (x < min):
                    min = x
                    min_label = j[1]
            if (i[1] != min_label):
                train1.remove(i)
                p = 1
        for i in train2:
            min = 10000000
            min_label = train1[0][1]
            for j in train1 + train3:
                x = distance(i[0], [j[0].height, j[0].weight, j[0].shoes])
                if (x < min):
                    min = x
                    min_label = j[1]
            if (i[1] != min_label):
                train2.remove(i)
                p = 1
        for i in train3:
            min = 10000000
            min_label = train1[0][1]
            for j in train1 + train2:
                x = distance(i[0], [j[0].height, j[0].weight, j[0].shoes])
                if (x < min):
                    min = x
                    min_label = j[1]
            if (i[1] != min_label):
                train3.remove(i)
                p = 1
        print(len(train1) + len(train2) + len(train3))
        ax_list[q].scatter([float(e[0].height) for e in (train1 + train2 + train3) if e[1] == 1],
                           [float(e[0].weight) for e in (train1 + train2 + train3) if e[1] == 1],
                           [float(e[0].shoes) for e in (train1 + train2 + train3) if e[1] == 1], color='blue')
        ax_list[q].scatter([float(e[0].height) for e in (train1 + train2 + train3) if e[1] == 0],
                           [float(e[0].weight) for e in (train1 + train2 + train3) if e[1] == 0],
                           [float(e[0].shoes) for e in (train1 + train2 + train3) if e[1] == 0], color='red')
        q += 1
    ax_list[q].scatter([float(e[0].height) for e in (train1 + train2 + train3) if e[1] == 1],
                       [float(e[0].weight) for e in (train1 + train2 + train3) if e[1] == 1],
                       [float(e[0].shoes) for e in (train1 + train2 + train3) if e[1] == 1], color='blue')
    ax_list[q].scatter([float(e[0].height) for e in (train1 + train2 + train3) if e[1] == 0],
                       [float(e[0].weight) for e in (train1 + train2 + train3) if e[1] == 0],
                       [float(e[0].shoes) for e in (train1 + train2 + train3) if e[1] == 0], color='red')
    train = train1 + train2 + train3
    boytrain, girltrain = [], []
    for i in train:
        if (i[1] == 1):
            boytrain.append(i)
        else:
            girltrain.append(i)
    boytrain_cut = [i[0] for i in boytrain]
    girltrain_cut = [i[0] for i in girltrain]
    return boytrain_cut, girltrain_cut


def knn_zip(boytrain, girltrain):
    fig = plt.figure()
    ax1, ax2, ax3, ax4 = fig.add_subplot(221, projection='3d'), fig.add_subplot(222, projection='3d'), fig.add_subplot(
        223, projection='3d'), fig.add_subplot(224, projection='3d')
    ax_list = [ax1, ax2, ax3, ax4]
    train = []
    for i in boytrain:
        train.append([i, 1])
    for j in girltrain:
        train.append([j, 0])
    random.shuffle(train)
    train_len = len(train)
    print(train_len)
    a, b = [], []
    b = train.copy()
    # print(b)
    p = b.pop(0)
    a.append(p)
    q = 1
    r = 0
    while (q):
        q = 0
        for i in b:
            # print(i)
            min = 1000000
            min_label = b[0][1]
            for j in a:
                x = distance(i[0], [float(j[0].height), float(j[0].weight), float(j[0].shoes)])
                if x < min:
                    min = x
                    min_label = j[1]
            if(i[1] != min_label):
                print(i)
                a.append(i)
                b.remove(i)
                q = 1
        ax_list[r].set_title('sample_num = ' + str(len(a)))
        ax_list[r].scatter([float(e[0].height) for e in a if e[1] == 1],
                           [float(e[0].weight) for e in a if e[1] == 1],
                           [float(e[0].shoes) for e in a if e[1] == 1], color='blue')
        ax_list[r].scatter([float(e[0].height) for e in a if e[1] == 0],
                            [float(e[0].weight) for e in a if e[1] == 0],
                            [float(e[0].shoes) for e in a if e[1] == 0], color='red')
        r += 1
        print(r)
        print(len(a))
    ax_list[r].scatter([float(e[0].height) for e in a if e[1] == 1],
                       [float(e[0].weight) for e in a if e[1] == 1],
                       [float(e[0].shoes) for e in a if e[1] == 1], color='blue')
    ax_list[r].scatter([float(e[0].height) for e in a if e[1] == 0],
                       [float(e[0].weight) for e in a if e[1] == 0],
                       [float(e[0].shoes) for e in a if e[1] == 0], color='red')
    boytrain, girltrain = [], []
    for i in a:
        if (i[1] == 1):
            boytrain.append(i)
        else:
            girltrain.append(i)
    boytrain_zip = [i[0] for i in boytrain]
    girltrain_zip = [i[0] for i in girltrain]
    return boytrain_zip, girltrain_zip



if __name__ == '__main__':
    # boytrain_cut, girltrain_cut = knn_cut(boytrain, girltrain)
    # boytrain_zip, girltrain_zip = knn_zip(boytrain, girltrain)
    res = knn_predict(5, boytest, girltest, boytrain, girltrain)
    res_only = knn_predict_only(5, boytrain, girltrain)
    correct_rate = correct(res)
    correct_rate_only = correct(res_only)
    roc_plot()
    # scatter_contour()
