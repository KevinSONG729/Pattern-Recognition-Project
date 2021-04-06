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
boytrain_index = random.sample(range(0, boycount), round(0.5 * boycount))
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


def train():
    boy_height_mean = np.mean(np.array([float(i.height) for i in boytrain]))
    boy_height_var = np.var([float(i.height) for i in boytrain])
    boy_weight_mean = np.mean(np.array([float(i.weight) for i in boytrain]))
    boy_weight_var = np.var([float(i.weight) for i in boytrain])
    boy_shoes_mean = np.mean(np.array([float(i.shoes) for i in boytrain]))
    boy_shoes_var = np.var([float(i.shoes) for i in boytrain])
    girl_height_mean = np.mean(np.array([float(i.height) for i in girltrain]))
    girl_height_var = np.var([float(i.height) for i in girltrain])
    girl_weight_mean = np.mean(np.array([float(i.weight) for i in girltrain]))
    girl_weight_var = np.var([float(i.weight) for i in girltrain])
    girl_shoes_mean = np.mean(np.array([float(i.shoes) for i in girltrain]))
    girl_shoes_var = np.var([float(i.shoes) for i in girltrain])
    boy_cov = np.cov([[float(i.height) for i in boytrain], [float(i.weight) for i in boytrain]])
    girl_cov = np.cov([[float(i.height) for i in girltrain], [float(i.weight) for i in girltrain]])
    boy_cov_3d = np.cov(
        [[float(i.height) for i in boytrain], [float(i.weight) for i in boytrain], [float(i.shoes) for i in boytrain]])
    girl_cov_3d = np.cov([[float(i.height) for i in girltrain], [float(i.weight) for i in girltrain],
                          [float(i.shoes) for i in girltrain]])

    return boy_height_mean, boy_height_var, boy_weight_mean, boy_weight_var, boy_shoes_mean, boy_shoes_var, \
           girl_height_mean, girl_height_var, girl_weight_mean, girl_weight_var, girl_shoes_mean, girl_shoes_var, boy_cov, girl_cov, boy_cov_3d, girl_cov_3d


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


def plot_norm():
    u1, u2, u3, u4 = boy_height_mean, boy_weight_mean, girl_height_mean, girl_weight_mean
    sigma1, sigma2, sigma3, sigma4 = np.sqrt(boy_height_var), np.sqrt(boy_weight_var), np.sqrt(
        girl_height_var), np.sqrt(girl_weight_var)
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224)
    pri_list = [[0.5, 0.5], [0.75, 0.25], [0.9, 0.1], [(len(boytrain) / (len(boytrain) + len(girltrain))),
                                                       (len(girltrain) / (len(boytrain) + len(girltrain)))]]
    marker_list = ['o', 'x', '+', '^']
    for p in range(0, 4):
        for i in [[u1, sigma1], [u2, sigma2], [u3, sigma3], [u4, sigma4]]:
            x = np.linspace(i[0] - 3 * i[1], i[0] + 3 * i[1], 50)
            y = st.norm.pdf(x, i[0], i[1])
            if i == [u1, sigma1]:
                yy = y * pri_list[p][0] / (y * pri_list[p][0] + st.norm.pdf(x, u3, sigma3) * pri_list[p][1])
                ax1.plot(x, y, color='Cyan')
                ax4.plot(x, yy, color='Cyan', marker=marker_list[p], markersize=7)
            elif i == [u2, sigma2]:
                yy = y * pri_list[p][0] / (y * pri_list[p][0] + st.norm.pdf(x, u4, sigma4) * pri_list[p][1])
                ax1.plot(x, y, color='#3366FF')
                ax4.plot(x, yy, color='#3366FF', marker=marker_list[p], markersize=7)
            elif i == [u3, sigma3]:
                yy = y * pri_list[p][0] / (y * pri_list[p][0] + st.norm.pdf(x, u1, sigma1) * pri_list[p][1])
                ax1.plot(x, y, color='#FF99CC')
                ax4.plot(x, yy, color='#FF99CC', marker=marker_list[p], markersize=7)
            else:
                yy = y * pri_list[p][0] / (y * pri_list[p][0] + st.norm.pdf(x, u2, sigma2) * pri_list[p][1])
                ax1.plot(x, y, color='#CC99FF')
                ax4.plot(x, yy, color='#CC99FF', marker=marker_list[p], markersize=7)
    ax1.legend(['boy_height', 'boy_weight', 'girl_height', 'girl_weight'])
    ax4.legend(['boy_height', 'boy_weight', 'girl_height', 'girl_weight'])
    miu1 = np.matrix([boy_height_mean, boy_weight_mean]).reshape(-1, 1)
    miu2 = np.matrix([girl_height_mean, girl_weight_mean]).reshape(-1, 1)
    size = 1
    for p in [boy_cov, girl_cov]:
        if (p == boy_cov).any():
            x = np.linspace(u1 - size * sigma1, u1 + size * sigma1, 100)
            y = np.linspace(u2 - size * sigma2, u2 + size * sigma2, 100)
        else:
            x = np.linspace(u3 - size * sigma3, u3 + size * sigma3, 100)
            y = np.linspace(u4 - size * sigma4, u4 + size * sigma4, 100)
        X, Y = np.meshgrid(x, y)
        Z1 = []
        Z2 = []
        for i in range(0, 100):
            for j in range(0, 100):
                X_feature = np.matrix([X[i, j], Y[i, j]]).reshape(-1, 1)
                if (p == boy_cov).any():
                    Z1.append(calc_norm(X_feature, miu1, boy_cov, 2))
                else:
                    Z2.append(calc_norm(X_feature, miu2, girl_cov, 2))
        if (p == boy_cov).any():
            surf1 = ax2.plot_surface(X, Y, np.array(Z1).reshape(100, 100), cmap=plt.get_cmap('coolwarm'))
            ax2.contour(X, Y, np.array(Z1).reshape(100, 100), zdim='z', offset=-0.0005, cmap='spring')
            fig.colorbar(surf1, shrink=0.5, aspect=5, ax=ax2)
            ax2.set_title('男生身高体重二维正态概率密度', FontProperties=fontstyle)
            ax2.set_xlabel('height/cm')
            ax2.set_ylabel('weight/kg')
        else:
            surf2 = ax3.plot_surface(X, Y, np.array(Z2).reshape(100, 100), cmap=plt.get_cmap('winter'))
            ax3.contour(X, Y, np.array(Z2).reshape(100, 100), zdim='z', offset=-0.0005, cmap='summer')
            fig.colorbar(surf2, shrink=0.5, aspect=5, ax=ax3)
            ax3.set_title('女生身高体重二维正态概率密度', FontProperties=fontstyle)
            ax3.set_xlabel('height/cm')
            ax3.set_ylabel('weight/kg')
    plt.grid()
    plt.show()


def predict(bp, gp):
    test_hei_result = []
    test_wei_result = []
    test_sho_result = []
    test_both_result = []
    test_all_result = []
    hei_yes = 0
    wei_yes = 0
    sho_yes = 0
    both_yes = 0
    all_yes = 0
    for i in boytest:
        zh = st.norm.pdf(float(i.height), boy_height_mean, np.sqrt(boy_height_var)) * bp - st.norm.pdf(
            float(i.height), girl_height_mean, np.sqrt(girl_height_var)) * gp
        zw = st.norm.pdf(float(i.weight), boy_weight_mean, np.sqrt(boy_weight_var)) * bp - st.norm.pdf(
            float(i.weight), girl_weight_mean, np.sqrt(girl_weight_var)) * gp
        zs = st.norm.pdf(float(i.shoes), boy_shoes_mean, np.sqrt(boy_shoes_var)) * bp - st.norm.pdf(
            float(i.shoes), girl_shoes_mean, np.sqrt(girl_shoes_var)) * gp
        zb = calc_norm(np.matrix([float(i.height), float(i.weight)]).reshape(-1, 1),
                       np.matrix([boy_height_mean, boy_weight_mean]).reshape(-1, 1), boy_cov, 2) * bp \
             - calc_norm(np.matrix([float(i.height), float(i.weight)]).reshape(-1, 1),
                         np.matrix([girl_height_mean, girl_weight_mean]).reshape(-1, 1), girl_cov, 2) * gp
        za = calc_norm(np.matrix([float(i.height), float(i.weight), float(i.shoes)]).reshape(-1, 1),
                       np.matrix([boy_height_mean, boy_weight_mean, boy_shoes_mean]).reshape(-1, 1), boy_cov_3d,
                       3) * bp - \
             calc_norm(np.matrix([float(i.height), float(i.weight), float(i.shoes)]).reshape(-1, 1),
                       np.matrix([girl_height_mean, girl_weight_mean, girl_shoes_mean]).reshape(-1, 1), girl_cov_3d,
                       3) * gp
        test_hei_result.append([zh, 1])
        test_wei_result.append([zw, 1])
        test_sho_result.append([zs, 1])
        test_both_result.append([zb, 1])
        test_all_result.append([za, 1])
    for i in girltest:
        zh = st.norm.pdf(float(i.height), boy_height_mean, np.sqrt(boy_height_var)) * bp - st.norm.pdf(
            float(i.height), girl_height_mean, np.sqrt(girl_height_var)) * gp
        zw = st.norm.pdf(float(i.weight), boy_weight_mean, np.sqrt(boy_weight_var)) * bp - st.norm.pdf(
            float(i.weight), girl_weight_mean, np.sqrt(girl_weight_var)) * gp
        zs = st.norm.pdf(float(i.shoes), boy_shoes_mean, np.sqrt(boy_shoes_var)) * bp - st.norm.pdf(
            float(i.shoes), girl_shoes_mean, np.sqrt(girl_shoes_var)) * gp
        zb = calc_norm(np.matrix([float(i.height), float(i.weight)]).reshape(-1, 1),
                       np.matrix([boy_height_mean, boy_weight_mean]).reshape(-1, 1), boy_cov, 2) * bp \
             - calc_norm(np.matrix([float(i.height), float(i.weight)]).reshape(-1, 1),
                         np.matrix([girl_height_mean, girl_weight_mean]).reshape(-1, 1), girl_cov, 2) * gp
        za = calc_norm(np.matrix([float(i.height), float(i.weight), float(i.shoes)]).reshape(-1, 1),
                       np.matrix([boy_height_mean, boy_weight_mean, boy_shoes_mean]).reshape(-1, 1), boy_cov_3d,
                       3) * bp - \
             calc_norm(np.matrix([float(i.height), float(i.weight), float(i.shoes)]).reshape(-1, 1),
                       np.matrix([girl_height_mean, girl_weight_mean, girl_shoes_mean]).reshape(-1, 1), girl_cov_3d,
                       3) * gp
        test_hei_result.append([zh, 0])
        test_wei_result.append([zw, 0])
        test_sho_result.append([zs, 0])
        test_both_result.append([zb, 0])
        test_all_result.append([za, 0])
    for i in test_hei_result:
        if i[0] > 0 and i[1] == 1 or i[0] < 0 and i[1] == 0:
            hei_yes += 1
    for i in test_wei_result:
        if i[0] > 0 and i[1] == 1 or i[0] < 0 and i[1] == 0:
            wei_yes += 1
    for i in test_both_result:
        if i[0] > 0 and i[1] == 1 or i[0] < 0 and i[1] == 0:
            both_yes += 1
    for i in test_sho_result:
        if i[0] > 0 and i[1] == 1 or i[0] < 0 and i[1] == 0:
            sho_yes += 1
    for i in test_all_result:
        if i[0] > 0 and i[1] == 1 or i[0] < 0 and i[1] == 0:
            all_yes += 1
    hei_yes = hei_yes / len(test_hei_result)
    wei_yes = wei_yes / len(test_wei_result)
    both_yes = both_yes / len(test_both_result)
    sho_yes = sho_yes / len(test_sho_result)
    all_yes = all_yes / len(test_all_result)
    return test_hei_result, test_wei_result, test_sho_result, test_both_result, test_all_result, hei_yes, wei_yes, sho_yes, both_yes, all_yes


def scatter_contour():
    x = np.linspace(150, 190, 100)
    y = np.linspace(40, 80, 100)
    X, Y = np.meshgrid(x, y)
    print(type(X))
    Z = np.zeros(shape=(100, 100))
    for i in range(0, 100):
        for j in range(0, 100):
            Z[i, j] = calc_norm(np.matrix([float(X[i, j]), float(Y[i, j])]).reshape(-1, 1),
                                np.matrix([boy_height_mean, boy_weight_mean]).reshape(-1, 1), boy_cov, 2) * (
                              len(boytrain) / (len(boytrain) + len(girltrain))) \
                      - calc_norm(np.matrix([float(X[i, j]), float(Y[i, j])]).reshape(-1, 1),
                                  np.matrix([girl_height_mean, girl_weight_mean]).reshape(-1, 1), girl_cov, 2) * (
                              len(girltrain) / (len(boytrain) + len(girltrain)))
    plt.scatter([float(i.height) for i in boytest], [float(i.weight) for i in boytest], color='blue')
    plt.scatter([float(i.height) for i in girltest], [float(i.weight) for i in girltest], color='red')
    C = plt.contour(X, Y, Z)
    plt.clabel(C, inline=True, fontsize=10)
    xx = sp.Symbol('x')
    fx = math.log(1 / sp.sqrt(2 * math.pi * boy_height_var)) - (xx - boy_height_mean) * (xx - boy_height_mean) / (
            2 * boy_height_var) + math.log(float(
        len(boytrain) / (len(boytrain) + len(girltrain)))) - math.log(1 / sp.sqrt(2 * math.pi * girl_height_var)) + (
                 xx - girl_height_mean) * (xx - girl_height_mean) / (2 * girl_height_var) - math.log(float(
        len(girltrain) / (len(boytrain) + len(girltrain))))
    xx = sp.solve(fx, xx)
    xxx = sp.Symbol('x')
    fx = math.log(1 / sp.sqrt(2 * math.pi * boy_weight_var)) - (xxx - boy_weight_mean) * (xxx - boy_weight_mean) / (
            2 * boy_weight_var) + math.log(float(
        len(boytrain) / (len(boytrain) + len(girltrain)))) - math.log(1 / sp.sqrt(2 * math.pi * girl_weight_var)) + (
                 xxx - girl_weight_mean) * (xxx - girl_weight_mean) / (2 * girl_weight_var) - math.log(float(
        len(girltrain) / (len(boytrain) + len(girltrain))))
    xxx = sp.solve(fx, xxx)
    plt.vlines([i for i in xx if i > 100 and i < 200], 40, 80, 'r', '--')
    plt.hlines([i for i in xxx if i > 40 and i < 80], 150, 180, 'g', '--')
    plt.text(x=[i for i in xx if i > 100 and i < 200][0] + 1, y=80, s='height:' + str(xx[1]), color='r')
    plt.text(y=[i for i in xxx if i > 40 and i < 80][0] + 1, x=150, s='weight:' + str(xxx[1]), color='g')
    plt.text(x=155, y=77, s='height_accuracy:' + str(hei_yes))
    plt.text(x=155, y=75, s='weight_accuracy:' + str(wei_yes))
    plt.text(x=155, y=73, s='both_accuracy:' + str(both_yes))
    plt.grid()
    plt.show()


def scatter_contour_3d():
    size = 15
    p = np.linspace(150, 200, size)
    q = np.linspace(40, 100, size)
    X, Y = np.meshgrid(p, q)
    Z = np.zeros(shape=(size, size))
    x = sp.Symbol('x')
    i1 = 0
    j1 = 0
    for i in np.linspace(150, 200, size):
        j1 = 0
        for j in np.linspace(40, 100, size):
            fx = (- 0.5 * (sp.Matrix([[i], [j], [x]]) - sp.Matrix([[boy_height_mean], [boy_weight_mean],
                                                                   [boy_shoes_mean]])).T * (
                          sp.Matrix(boy_cov_3d) ** (-1)) * \
                  (sp.Matrix([[i], [j], [x]])
                   - sp.Matrix([[boy_height_mean], [boy_weight_mean],
                                [boy_shoes_mean]])) + sp.Matrix(
                        [sp.log(float(len(boytrain) / (len(boytrain) + len(girltrain))))]) - sp.Matrix([sp.log(
                        sp.sqrt(sp.Matrix(boy_cov_3d).det()))]) + \
                  0.5 * (sp.Matrix([[i], [j], [x]]) - sp.Matrix([[girl_height_mean], [girl_weight_mean],
                                                                 [girl_shoes_mean]])).T * (
                          sp.Matrix(girl_cov_3d) ** (-1)) * (sp.Matrix([[i], [j], [x]])
                                                             - sp.Matrix([[girl_height_mean], [girl_weight_mean],
                                                                          [girl_shoes_mean]])) - sp.Matrix(
                        [sp.log(float(len(girltrain) / (len(boytrain) + len(girltrain))))]) + sp.Matrix([sp.log(
                        sp.sqrt(sp.Matrix(girl_cov_3d).det()))]))[0]
            y = sp.solve(fx, x)
            # print(y)
            if type(y[1]) == sp.core.numbers.Add:
                Z[i1, j1] = sp.re(y[1])
            else:
                Z[i1, j1] = y[1]
            j1 += 1
        i1 += 1
    # print(Z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D([float(i.height) for i in boytest], [float(i.weight) for i in boytest],
                 [float(i.shoes) for i in boytest])
    ax.scatter3D([float(i.height) for i in girltest], [float(i.weight) for i in girltest],
                 [float(i.shoes) for i in girltest])
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm')
    fig.colorbar(surf, shrink=0.5, aspect=5, ax=ax)
    return X, Y, Z


def roc_plot():
    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax_list = [ax1, ax2, ax3, ax4, ax5]
    label_list = ['Height', 'Weight', 'ShoeSize', 'Height/Weight', 'Height/Weight/ShoeSize']
    pri_list = [[0.5, 0.5], [0.75, 0.25], [0.9, 0.1], [float(len(boytrain) / (len(boytrain) + len(girltrain))),
                                                       float(len(girltrain) / (len(boytrain) + len(girltrain)))]]
    color_list = ['#FF99CC', 'red', 'green', '#3366FF']
    for p in range(0, 4):
        test_hei_result, test_wei_result, test_sho_result, test_both_result, test_all_result, hei_yes, wei_yes, sho_yes, both_yes, all_yes = predict(
            pri_list[p][0],
            pri_list[p][1])
        test_hei_result_sorted, test_wei_result_sorted, test_sho_result_sorted, test_both_result_sorted, test_all_result_sorted \
            = sorted(test_hei_result, reverse=True), \
              sorted(test_wei_result, reverse=True), \
              sorted(test_sho_result, reverse=True), \
              sorted(test_both_result, reverse=True), \
              sorted(test_all_result, reverse=True)
        for i in [test_hei_result_sorted, test_wei_result_sorted, test_sho_result_sorted, test_both_result_sorted,
                  test_all_result_sorted]:
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
            if i == test_hei_result_sorted:
                ax1.plot(x, y, color=color_list[p])
            elif i == test_wei_result_sorted:
                ax2.plot(x, y, color=color_list[p])
            elif i == test_sho_result_sorted:
                ax3.plot(x, y, color=color_list[p])
            elif i == test_both_result_sorted:
                ax4.plot(x, y, color=color_list[p])
            else:
                ax5.plot(x, y, color=color_list[p])
    for i in range(0, 5):
        ax_list[i].set_xlabel = 'FPR'
        ax_list[i].set_ylabel = 'TPR'
        ax_list[i].set_title(label_list[i])
    fig.legend(['0.5/0.5', '0.75/0.25', '0.9/0.1', 'boypri/girlpri'])


def Parzen():
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax_list = [ax1, ax2, ax3, ax4]
    var_list = [0.09, 0.25, 1, 4]
    hn_title_list = ['hn = 0.09', 'hn = 0.25', 'hn = 1', 'hn = 4']
    label_list = ['Boy_Height', 'Boy_Weight', 'Boy_ShoeSize', 'Girl_Height', 'Girl_Weight', 'Girl_ShoeSize']
    for i in range(0, 4):
        height, weight, shoe = [], [], []
        hheight, wweight, sshoe = [], [], []
        for j in np.linspace(30, 200, 100):
            sumh, sumw, sums = 0, 0, 0
            ssumh, ssumw, ssums = 0, 0, 0
            for k in boytrain:
                sumh += norm.pdf((j - float(k.height)) / np.sqrt(var_list[i]), 0, 1)
                sumw += norm.pdf((j - float(k.weight)) / np.sqrt(var_list[i]), 0, 1)
                sums += norm.pdf((j - float(k.shoes)) / np.sqrt(var_list[i]), 0, 1)
            sumh /= (len(boytrain))
            sumw /= (len(boytrain))
            sums /= (len(boytrain))
            height.append(sumh)
            weight.append(sumw)
            shoe.append(sums)
            for k in girltrain:
                ssumh += norm.pdf((j - float(k.height)) / np.sqrt(var_list[i]), 0, 1)
                ssumw += norm.pdf((j - float(k.weight)) / np.sqrt(var_list[i]), 0, 1)
                ssums += norm.pdf((j - float(k.shoes)) / np.sqrt(var_list[i]), 0, 1)
            ssumh /= (len(girltrain))
            ssumw /= (len(girltrain))
            ssums /= (len(girltrain))
            hheight.append(ssumh)
            wweight.append(ssumw)
            sshoe.append(ssums)
        ax_list[i].plot(np.linspace(40, 200, 100), height)
        ax_list[i].plot(np.linspace(40, 200, 100), weight)
        ax_list[i].plot(np.linspace(40, 200, 100), shoe)
        ax_list[i].plot(np.linspace(40, 200, 100), hheight)
        ax_list[i].plot(np.linspace(40, 200, 100), wweight)
        ax_list[i].plot(np.linspace(40, 200, 100), sshoe)
        ax_list[i].set_title(hn_title_list[i])
    fig.legend(label_list)


def Parzen_predict(bp, gp):
    test_hei_resultp = []
    test_wei_resultp = []
    test_sho_resultp = []
    test_both_resultp = []
    test_all_resultp = []
    hei_yesp = 0
    wei_yesp = 0
    sho_yesp = 0
    both_yesp = 0
    all_yesp = 0
    var_list = [0.09, 0.25, 1, 4]

    for i in boytest:
        zh = float(sum([norm.pdf((float(i.height) - float(p.height)) / np.sqrt(0.09), 0, 1) for p in boytrain]) / len(
            boytrain)) * bp - float(sum(
            [norm.pdf((float(i.height) - float(p.height)) / np.sqrt(0.09), 0, 1) for p in girltrain]) / len(
            girltrain)) * gp
        zw = float(sum([norm.pdf((float(i.weight) - float(p.weight)) / np.sqrt(0.09), 0, 1) for p in boytrain]) / len(
            boytrain)) * bp - float(sum(
            [norm.pdf((float(i.weight) - float(p.weight)) / np.sqrt(0.09), 0, 1) for p in girltrain]) / len(
            girltrain)) * gp
        zs = float(
            sum([norm.pdf((float(i.shoes) - float(p.shoes)) / np.sqrt(0.09), 0, 1) for p in boytrain]) / len(
                boytrain)) * bp - float(
            sum(
                [norm.pdf((float(i.shoes) - float(p.shoes)) / np.sqrt(0.09), 0, 1) for p in girltrain]) / len(
                girltrain)) * gp
        zb = float(sum(calc_norm(np.matrix([float(i.height), float(i.weight)]).reshape(-1, 1),
                                 np.matrix([float(p.height), float(p.weight)]).reshape(-1, 1), boy_cov * 0.09, 2) for p
                       in
                       boytrain) / len(boytrain)) * bp \
             - float(sum(calc_norm(np.matrix([float(i.height), float(i.weight)]).reshape(-1, 1),
                                   np.matrix([float(p.height), float(p.weight)]).reshape(-1, 1), girl_cov * 0.09, 2) for
                         p in
                         girltrain) / len(girltrain)) * gp
        za = float(sum(calc_norm(np.matrix([float(i.height), float(i.weight), float(i.shoes)]).reshape(-1, 1),
                                 np.matrix([float(p.height), float(p.weight), float(p.shoes)]).reshape(-1, 1),
                                 boy_cov_3d * 0.09,
                                 3) for p in boytrain) / len(boytrain)) * bp - \
             float(sum(calc_norm(np.matrix([float(i.height), float(i.weight), float(i.shoes)]).reshape(-1, 1),
                                 np.matrix([float(p.height), float(p.weight), float(p.shoes)]).reshape(-1, 1),
                                 girl_cov_3d * 0.09,
                                 3) for p in girltrain) / len(girltrain)) * gp
        test_hei_resultp.append([zh, 1])
        test_wei_resultp.append([zw, 1])
        test_sho_resultp.append([zs, 1])
        test_both_resultp.append([zb, 1])
        test_all_resultp.append([za, 1])
    for i in girltest:
        zh = float(sum([norm.pdf((float(i.height) - float(p.height)) / np.sqrt(0.09), 0, 1) for p in boytrain]) / len(
            boytrain)) * bp - float(sum(
            [norm.pdf((float(i.height) - float(p.height)) / np.sqrt(0.09), 0, 1) for p in girltrain]) / len(
            girltrain)) * gp
        zw = float(sum([norm.pdf((float(i.weight) - float(p.weight)) / np.sqrt(0.09), 0, 1) for p in boytrain]) / len(
            boytrain)) * bp - float(sum(
            [norm.pdf((float(i.weight) - float(p.weight)) / np.sqrt(0.09), 0, 1) for p in girltrain]) / len(
            girltrain)) * gp
        zs = float(
            sum([norm.pdf((float(i.shoes) - float(p.shoes)) / np.sqrt(0.09), 0, 1) for p in boytrain]) / len(
                boytrain)) * bp - float(
            sum(
                [norm.pdf((float(i.shoes) - float(p.shoes)) / np.sqrt(0.09), 0, 1) for p in girltrain]) / len(
                girltrain)) * gp
        zb = float(sum(calc_norm(np.matrix([float(i.height), float(i.weight)]).reshape(-1, 1),
                                 np.matrix([float(p.height), float(p.weight)]).reshape(-1, 1), boy_cov * 0.09, 2) for p
                       in
                       boytrain) / len(boytrain)) * bp \
             - float(sum(calc_norm(np.matrix([float(i.height), float(i.weight)]).reshape(-1, 1),
                                   np.matrix([float(p.height), float(p.weight)]).reshape(-1, 1), girl_cov * 0.09, 2) for
                         p in
                         girltrain) / len(girltrain)) * gp
        za = float(sum(calc_norm(np.matrix([float(i.height), float(i.weight), float(i.shoes)]).reshape(-1, 1),
                                 np.matrix([float(p.height), float(p.weight), float(p.shoes)]).reshape(-1, 1),
                                 boy_cov_3d * 0.09,
                                 3) for p in boytrain) / len(boytrain)) * bp - \
             float(sum(calc_norm(np.matrix([float(i.height), float(i.weight), float(i.shoes)]).reshape(-1, 1),
                                 np.matrix([float(p.height), float(p.weight), float(p.shoes)]).reshape(-1, 1),
                                 girl_cov_3d * 0.09,
                                 3) for p in girltrain) / len(girltrain)) * gp
        test_hei_resultp.append([zh, 0])
        test_wei_resultp.append([zw, 0])
        test_sho_resultp.append([zs, 0])
        test_both_resultp.append([zb, 0])
        test_all_resultp.append([za, 0])
    for i in test_hei_resultp:
        if i[0] > 0 and i[1] == 1 or i[0] < 0 and i[1] == 0:
            hei_yesp += 1
    for i in test_wei_resultp:
        if i[0] > 0 and i[1] == 1 or i[0] < 0 and i[1] == 0:
            wei_yesp += 1
    for i in test_both_resultp:
        if i[0] > 0 and i[1] == 1 or i[0] < 0 and i[1] == 0:
            both_yesp += 1
    for i in test_sho_resultp:
        if i[0] > 0 and i[1] == 1 or i[0] < 0 and i[1] == 0:
            sho_yesp += 1
    for i in test_all_resultp:
        if i[0] > 0 and i[1] == 1 or i[0] < 0 and i[1] == 0:
            all_yesp += 1
    hei_yesp = hei_yesp / len(test_hei_result)
    wei_yesp = wei_yesp / len(test_wei_result)
    both_yesp = both_yesp / len(test_both_result)
    sho_yesp = sho_yesp / len(test_sho_result)
    all_yesp = all_yesp / len(test_all_result)
    return test_hei_resultp, test_wei_resultp, test_sho_resultp, test_both_resultp, test_all_resultp, hei_yesp, wei_yesp, sho_yesp, both_yesp, all_yesp


if __name__ == '__main__':
    boy_height_mean, boy_height_var, boy_weight_mean, boy_weight_var, boy_shoes_mean, boy_shoes_var, \
    girl_height_mean, girl_height_var, girl_weight_mean, girl_weight_var, girl_shoes_mean, girl_shoes_var, boy_cov, girl_cov, boy_cov_3d, girl_cov_3d = train()
    # plot_norm()
    test_hei_result, test_wei_result, test_sho_result, test_both_result, test_all_result, hei_yes, wei_yes, sho_yes, both_yes, all_yes = predict(
        float(len(boytrain) / (len(boytrain) + len(girltrain))),
        float(len(girltrain) / (len(boytrain) + len(girltrain))))
    scatter_contour()
    # X, Y, Z = scatter_contour_3d()
    # roc_plot()
    # Parzen()
    '''test_hei_resultp, test_wei_resultp, test_sho_resultp, test_both_resultp, test_all_resultp, hei_yesp, wei_yesp, sho_yesp, both_yesp, all_yesp = Parzen_predict(
        float(len(boytrain) / (len(boytrain) + len(girltrain))),
        float(len(girltrain) / (len(boytrain) + len(girltrain))))'''
