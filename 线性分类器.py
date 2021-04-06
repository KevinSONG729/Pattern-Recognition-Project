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
from matplotlib.pyplot import MultipleLocator

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
boytrain_index = random.sample(range(0, boycount), round(0.8 * boycount))
boytest_index = list(set(range(0, boycount)).difference(set(boytrain_index)))
girltrain_index = random.sample(range(0, girlcount), round(0.8 * girlcount))
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

boypri = float(len(boytrain) / (len(boytrain) + len(girltrain)))
girlpri = float(len(girltrain) / (len(boytrain) + len(girltrain)))
boy_cov = np.cov([[float(i.height) for i in boytrain], [float(i.weight) for i in boytrain]])
girl_cov = np.cov([[float(i.height) for i in girltrain], [float(i.weight) for i in girltrain]])


def Fisher_gx(w, x, m1, m2, boypri, girlpri, w0):
    # return float(np.transpose(w) * (x - 0.5 * (m1 + m2)) - np.log(float(girlpri / boypri)))-0.760
    return np.transpose(w) * x - w0


def Fisher(boytrain, girltrain, boytest, girltest):
    m1, m2 = [], []
    s1, s2, s = np.zeros(shape=(2, 2)), np.zeros(shape=(2, 2)), np.zeros(shape=(2, 2))
    boypri = float(len(boytrain) / (len(boytrain) + len(girltrain)))
    girlpri = float(len(girltrain) / (len(boytrain) + len(girltrain)))
    m1.append(np.mean([float(i.height) for i in boytrain]))
    m1.append(np.mean([float(i.weight) for i in boytrain]))
    m2.append(np.mean([float(i.height) for i in girltrain]))
    m2.append(np.mean([float(i.weight) for i in girltrain]))
    for i in [boytrain, girltrain]:
        for j in i:
            if i == boytrain:
                a = np.matrix([float(j.height), float(j.weight)]).reshape(-1, 1) - np.matrix(m1).reshape(-1, 1)
                s1 += a * np.transpose(a)
            else:
                a = np.matrix([float(j.height), float(j.weight)]).reshape(-1, 1) - np.matrix(m2).reshape(-1, 1)
                s2 += a * np.transpose(a)
    s = s1 + s2
    w = np.linalg.inv(s) * (np.matrix(m1).reshape(-1, 1) - np.matrix(m2).reshape(-1, 1))
    m1_t = np.transpose(w) * np.matrix(m1).reshape(-1, 1)
    m2_t = np.transpose(w) * np.matrix(m2).reshape(-1, 1)

    w0 = (m1_t + m2_t) / 2 + np.log(boypri / girlpri) / (float(len(boytrain) + len(girltrain)) - 2)
    '''-0.5 * np.transpose(np.matrix(m1).reshape(-1, 1) + np.matrix(m2).reshape(-1, 1)) * np.linalg.inv(s) * (
            np.matrix(m1).reshape(-1, 1) - np.matrix(m2).reshape(-1, 1)) - np.log(girlpri / boypri)'''
    res = []
    res_bayes = []
    for i in boytest:
        res.append([
            Fisher_gx(w, np.matrix([float(i.height), float(i.weight)]).reshape(-1, 1), np.matrix(m1).reshape(-1, 1),
                      np.matrix(m2).reshape(-1, 1), boypri, girlpri, w0), 1])
        res_bayes.append([float(bayes_2(float(i.height), float(i.weight))), 1])
    for i in girltest:
        res.append([
            Fisher_gx(w, np.matrix([float(i.height), float(i.weight)]).reshape(-1, 1), np.matrix(m1).reshape(-1, 1),
                      np.matrix(m2).reshape(-1, 1), boypri, girlpri, w0), 0])
        res_bayes.append([float(bayes_2(float(i.height), float(i.weight))), 0])
    return res, res_bayes, w, m1, m2, w0


def correct(res, res_bayes, boytest, girltest):
    correct_rate = 0
    correct_rate_bayes = 0
    for i in res:
        if (i[0] > 0 and i[1] == 1 or i[0] < 0 and i[1] == 0):
            correct_rate += 1
    for i in res_bayes:
        if (i[0] > 0 and i[1] == 1 or i[0] < 0 and i[1] == 0):
            correct_rate_bayes += 1
    correct_rate /= (len(boytest) + len(girltest))
    correct_rate_bayes /= (len(boytest) + len(girltest))
    return correct_rate, correct_rate_bayes


def calc_norm(X_vector, miu_vector, cov, d):
    if d == 2:
        z = 1 / 2 * math.pi * np.sqrt(np.linalg.det(cov)) * np.exp(
            -0.5 * np.transpose(X_vector - miu_vector) * np.linalg.inv(cov) * (
                    X_vector - miu_vector))
    else:
        z = 1 / math.pow(2 * math.pi, 0.5 * d) * np.sqrt(np.linalg.det(cov)) * np.exp(
            -0.5 * np.transpose(X_vector - miu_vector) * np.linalg.inv(cov) * (
                    X_vector - miu_vector))
    return z


def bayes_2(x, y):
    a = calc_norm(np.matrix([float(x), float(y)]).reshape(-1, 1),
                  np.matrix([np.mean(np.array([float(i.height) for i in boytrain])),
                             np.mean(np.array([float(i.weight) for i in boytrain]))]).reshape(-1, 1), boy_cov, 2) * (
            boypri) \
        - calc_norm(np.matrix([float(x), float(y)]).reshape(-1, 1),
                    np.matrix([np.mean(np.array([float(i.height) for i in girltrain])),
                               np.mean(np.array([float(i.weight) for i in girltrain]))]).reshape(-1, 1), girl_cov,
                    2) * (
            girlpri)
    return a


def Fisher_plot(w, m1, m2, w0):
    x = np.linspace(140, 190, 100)
    y = np.linspace(40, 90, 100)
    X, Y = np.meshgrid(x, y)
    Z1, Z2 = np.zeros(shape=(100, 100)), np.zeros(shape=(100, 100))
    a_b, a_g = [], []
    for i in range(0, 100):
        for j in range(0, 100):
            Z1[i][j] = Fisher_gx(w, np.matrix([x[i], y[j]]).reshape(-1, 1), m1, m2, boypri, girlpri, w0)
            Z2[i][j] = bayes_2(x[i], y[j])
    k = float(w[0][0] / w[1][0])
    for i in [boytest, girltest]:
        for j in i:
            if i == boytest:
                a_b.append((float(j.height) + k * float(j.weight)) / (k * k + 1))
            else:
                a_g.append((float(j.height) + k * float(j.weight)) / (k * k + 1))
    x_major_locator = MultipleLocator(10)
    y_major_locator = MultipleLocator(10)
    C1 = plt.contour(X, Y, Z1, 0, color='#3366FF')
    C2 = plt.contour(X, Y, Z2, 0, color='#FF99CC')
    plt.scatter([float(i.height) for i in boytest], [float(i.weight) for i in boytest], color='blue')
    plt.scatter([float(i.height) for i in girltest], [float(i.weight) for i in girltest], color='red')
    plt.scatter([i for i in a_b], [k * i + w0 / k for i in a_b], color='blue')
    plt.scatter([i for i in a_g], [k * i + w0 / k for i in a_g], color='red')
    plt.clabel(C1, inline=True, fontsize=10)
    plt.clabel(C2, inline=True, fontsize=10)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.legend(['boy', 'girl'])
    plt.show()


def Fisher_only_one():
    boytrain_len = len(boytrain)
    print(boytrain_len)
    girltrain_len = len(girltrain)
    correct_rate_only = 0
    for i in reversed(range(0, boytrain_len - 1)):
        boy_temp = boytrain.copy()
        p = boy_temp.pop(i)
        boypri = len(boy_temp) / (len(boy_temp) + len(girltrain))
        girlpri = len(girltrain) / (len(boy_temp) + len(girltrain))
        res, res_bayes, w, m1, m2, w0 = Fisher(boy_temp, girltrain, boytest, girltest)
        x = Fisher_gx(w, np.matrix([float(p.height), float(p.weight)]).reshape(-1, 1),
                      np.matrix(m1).reshape(-1, 1), np.matrix(m2).reshape(-1, 1), boypri, girlpri, w0)
        # print(x)
        if (x > 0):
            correct_rate_only += 1
    for i in reversed(range(0, girltrain_len - 1)):
        girl_temp = girltrain.copy()
        p = girl_temp.pop(i)
        boypri = len(boytrain) / (len(boytrain) + len(girl_temp))
        girlpri = len(girl_temp) / (len(boytrain) + len(girl_temp))
        res, res_bayes, w, m1, m2, w0 = Fisher(boytrain, girl_temp, boytest, girltest)
        x = Fisher_gx(w, np.matrix([float(p.height), float(p.weight)]).reshape(-1, 1),
                      np.matrix(m1).reshape(-1, 1), np.matrix(m2).reshape(-1, 1), boypri, girlpri, w0)
        # print(x)
        if (x < 0):
            correct_rate_only += 1
    correct_rate_only /= (boytrain_len + girltrain_len)
    return correct_rate_only

def roc_plot(res, res_bayes):
    res = sorted(res, reverse = True)
    res_bayes = sorted(res_bayes, reverse = True)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax_list = [ax1, ax2]
    label_list = ['Fisher', 'Bayes']
    for i in [res, res_bayes]:
        x, y = [], []
        xx, yy = 0, 0
        x.append(xx)
        y.append(yy)
        for j in i:
            if j[1] == 1:
                 yy += float(1 / len(boytest))
            else:
                 xx += float(1 / len(girltest))
            x.append(xx)
            y.append(yy)
        if i == res:
             ax1.plot(x, y)
        else:
             ax2.plot(x, y)
    for i in range(0, 2):
        ax_list[i].set_xlabel = 'FPR'
        ax_list[i].set_ylabel = 'TPR'
        ax_list[i].set_title(label_list[i])


if __name__ == '__main__':
    res, res_bayes, w, m1, m2, w0 = Fisher(boytrain, girltrain, boytest, girltest)
    correct_rate, correct_rate_bayes = correct(res, res_bayes, boytest, girltest)
    # Fisher_plot(w, np.matrix(m1).reshape(-1, 1), np.matrix(m2).reshape(-1, 1), w0)
    correct_rate_only = Fisher_only_one()
    # roc_plot(res, res_bayes)
