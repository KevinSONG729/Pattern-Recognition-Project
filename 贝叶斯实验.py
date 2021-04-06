import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import scipy.stats as st
import pandas as pd
import operator


class test_node:
    def __init__(self):
        self.height = 0
        self.weight = 0
        self.label = ""


class roc_node:
    def __init__(self):
        self.jus = 0
        self.label = ""


df = pd.read_excel("D:/专业课/模式识别/forstudent/实验数据/genderdata/周五实验课数据统计_55.xlsx")
df_test = pd.read_excel("D:/专业课/模式识别/forstudent/实验数据/genderdata/2018冬季班模式识别学生数据.xlsx")
f1 = open("D:/专业课/模式识别/forstudent/实验数据/genderdata/girl.txt")
f2 = open("D:/专业课/模式识别/forstudent/实验数据/genderdata/boy.txt")
data = df.values
data_test = df_test.values
man_height = []
woman_height = []
test_list = []
roc_list = []
roc_list_height = []
roc_list_weight = []
sex = []
for row in data:
    if row[1] == "男":
        man_height.append(row[2])
    else:
        woman_height.append(row[2])
    sex.append(row[1])

man_weight = []
woman_weight = []
for row in data:
    if row[1] == "男":
        man_weight.append(row[3])
    else:
        woman_weight.append(row[3])

manlen = len(man_height)
womanlen = len(woman_height)
manlen_test3 = 0
womanlen_test3 = 0

for row in data_test:
    a = test_node()
    if row[3] == "男":
        a.height = row[0]
        a.weight = row[1]
        a.label = "男"
        manlen_test3 += 1
        test_list.append(a)
    else:
        a.height = row[0]
        a.weight = row[1]
        a.label = "女"
        womanlen_test3 += 1
        test_list.append(a)
for row in f1.readlines():
    a = test_node()
    a.height = int(row.split()[0])
    a.weight = float(row.split()[1])
    a.label = "女"
    womanlen_test3 += 1
    test_list.append(a)
for row in f2.readlines():
    a = test_node()
    a.height = int(row.split()[0])
    a.weight = float(row.split()[1])
    a.label = "男"
    manlen_test3 += 1
    test_list.append(a)

man_height_mean, man_height_std = norm.fit(man_height)  # 男生升高分布参数
man_weight_mean, man_weight_std = norm.fit(man_weight)  # 男生体重分布参数
woman_height_mean, woman_height_std = norm.fit(woman_height)  # 女生升高分布参数
woman_weight_mean, woman_weight_std = norm.fit(woman_weight)  # 女生体重分布参数

man_height_variance = man_height_std ** 2
man_weight_variance = man_weight_std ** 2
woman_height_variance = woman_height_std ** 2
woman_weight_variance = woman_weight_std ** 2

man_height_mean_cntr, man_height_var_cntr, man_height_std_cntr = st.bayes_mvs(man_height)
man_weight_mean_cntr, man_weight_var_cntr, man_weight_std_cntr = st.bayes_mvs(man_weight)
woman_height_mean_cntr, woman_height_var_cntr, woman_height_std_cntr = st.bayes_mvs(woman_height)
woman_weight_mean_cntr, woman_weight_var_cntr, woman_weight_std_cntr = st.bayes_mvs(woman_weight)


# 求协方差矩阵
def get_covariance_matrix_coefficient(arr1, arr2):  # arr1与arr2长度相等
    datalength1 = len(arr1)
    datalength2 = len(arr2)
    sum_temp = []
    for i in range(datalength1):
        sum_temp.append((arr1[i] - sum(arr1) / datalength1) * (arr2[i] - sum(arr2) / datalength2))
        c12 = sum(sum_temp)
    covariance_matrix_c12 = c12 / (datalength1 - 1)
    return covariance_matrix_c12


man_c11 = man_height_variance
man_c22 = man_weight_variance
man_c12 = man_c21 = get_covariance_matrix_coefficient(man_height, man_weight)
man_covariance_matrix = np.matrix([[man_c11, man_c12], [man_c21, man_c22]])
woman_c11 = woman_height_variance
woman_c22 = woman_weight_variance
woman_c12 = woman_c21 = get_covariance_matrix_coefficient(woman_height, woman_weight)
woman_covariance_matrix = np.matrix([[woman_c11, woman_c12], [woman_c21, woman_c22]])

# 求男生、女生先验概率
man_priori_probability = manlen / (manlen + womanlen)
woman_priori_probability = 1 - man_priori_probability

# 平均特征向量(平均值由极大似然法给出)
man_feature_mean_vector = np.matrix([[man_height_mean], [man_weight_mean]])
woman_feature_mean_vector = np.matrix([[woman_height_mean], [woman_weight_mean]])


# 定义等高线高度函数(两个特征)
def f(sample_height, sample_weight):
    mytemp1 = np.zeros(shape=(100, 100))
    for i in range(100):
        for j in range(100):
            sample_vector = np.matrix([[sample_height[i, j]], [sample_weight[i, j]]])
            # sample_vector_T=np.transpose(sample_vector)
            # 定义决策函数
            mytemp1[i, j] = 0.5 * np.transpose(sample_vector - man_feature_mean_vector) * (
                np.linalg.inv(man_covariance_matrix)) * \
                            (sample_vector - man_feature_mean_vector) - 0.5 * np.transpose(
                sample_vector - woman_feature_mean_vector) * \
                            (np.linalg.inv(woman_covariance_matrix)) * (sample_vector - woman_feature_mean_vector) + \
                            0.5 * math.log(
                (np.linalg.det(man_covariance_matrix)) / (np.linalg.det(woman_covariance_matrix))) - \
                            math.log(man_priori_probability / woman_priori_probability)
    return mytemp1


# 定义决策线(一个特征)
def f_singlefeature(sample_feature, mode):
    mytemp1 = np.zeros((1, 100))
    for i in range(0, 100):
        if mode == "height":
            mytemp1[0][i] = 0.5 * (sample_feature[i] - man_height_mean) * \
                            (sample_feature[i] - man_height_mean) / man_height_variance - 0.5 * (
                                        sample_feature[i] - woman_height_mean) * \
                            (sample_feature[i] - woman_height_mean) / woman_height_variance + \
                            0.5 * math.log(man_height_variance / woman_height_variance) - \
                            math.log(man_priori_probability / woman_priori_probability)
        elif mode == "weight":
            mytemp1[0][i] = 0.5 * (sample_feature[i] - man_weight_mean) * \
                            (sample_feature[i] - man_weight_mean) / man_weight_variance - 0.5 * (
                                        sample_feature[i] - woman_weight_mean) * \
                            (sample_feature[i] - woman_weight_mean) / woman_weight_variance + \
                            0.5 * math.log(man_weight_variance / woman_weight_variance) - \
                            math.log(man_priori_probability / woman_priori_probability)
    return mytemp1


def Parzen_Estimate():
    len = manlen + womanlen
    a = []
    b = []
    for t in test_list:
        t1 = t.height
        t2 = t.weight
        p1 = 0
        p2 = 0
        for h in sorted(man_height):
            p1 += math.exp(-0.5 * pow(abs(h - t1) * math.sqrt(len) / 185, 2))
        for h in sorted(woman_height):
            p2 += math.exp(-0.5 * pow(abs(h - t2) * math.sqrt(len), 2))
        p1 *= 1 / math.sqrt(2 * math.pi * len * 185)
        p2 *= 1 / math.sqrt(2 * math.pi * len)
        a.append(p1)
        b.append(p2)
    return a, b


parzen_a, parzen_b = Parzen_Estimate()


# 求测试数据的判别值，获得roc列表
def f1(test_height, test_weight):
    sample_vector = np.matrix([[float(test_height)], [test_weight]])
    temp = 0.5 * np.transpose(sample_vector - man_feature_mean_vector) * (np.linalg.inv(man_covariance_matrix)) * \
           (sample_vector - man_feature_mean_vector) - 0.5 * np.transpose(sample_vector - woman_feature_mean_vector) * \
           (np.linalg.inv(woman_covariance_matrix)) * (sample_vector - woman_feature_mean_vector) + \
           0.5 * math.log((np.linalg.det(man_covariance_matrix)) / (np.linalg.det(woman_covariance_matrix))) - \
           math.log(man_priori_probability / woman_priori_probability)
    return temp


def f1_single(test_feature, mode):
    if mode == "height":
        temp = 0.5 * (test_feature - man_height_mean) * \
               (test_feature - man_height_mean) / man_height_variance - 0.5 * (test_feature - woman_height_mean) * \
               (test_feature - woman_height_mean) / woman_height_variance + \
               0.5 * math.log(man_height_variance / woman_height_variance) - \
               math.log(man_priori_probability / woman_priori_probability)
    elif mode == "weight":
        temp = 0.5 * (test_feature - man_weight_mean) * \
               (test_feature - man_weight_mean) / man_weight_variance - 0.5 * (test_feature - woman_weight_mean) * \
               (test_feature - woman_weight_mean) / woman_weight_variance + \
               0.5 * math.log(man_weight_variance / woman_weight_variance) - \
               math.log(man_priori_probability / woman_priori_probability)
    return temp


for node in test_list:
    x = f1(node.height, node.weight)
    x_height = f1_single(node.height, "height")
    x_weight = f1_single(node.weight, "weight")
    x1 = roc_node()
    x1.label = node.label
    x1.jus = x
    roc_list.append(x1)
    x1_height = roc_node()
    x1_height.label = node.label
    x1_height.jus = x_height
    roc_list_height.append(x1_height)
    x1_weight = roc_node()
    x1_weight.label = node.label
    x1_weight.jus = x_weight
    roc_list_weight.append(x1_weight)
cmpfun = operator.attrgetter("jus")
roc_list.sort(key=cmpfun)
roc_list_height.sort(key=cmpfun)
roc_list_weight.sort(key=cmpfun)
roc_list.reverse()
roc_list_height.reverse()
roc_list_weight.reverse()


# 画roc曲线(两个特征)
def roc(roc_list):
    roc_plot_x = []
    roc_plot_y = []
    roc_plotnode_x = 0
    roc_plotnode_y = 0
    error_num = 0
    roc_plot_x.append(roc_plotnode_x)
    roc_plot_y.append(roc_plotnode_y)
    for node in roc_list:
        if node.label == "男":
            roc_plotnode_y += 1 / manlen_test3
            roc_plot_x.append(roc_plotnode_x)
            roc_plot_y.append(roc_plotnode_y)
            if node.jus > 0:
                error_num += 1
        else:
            roc_plotnode_x += 1 / womanlen_test3
            roc_plot_x.append(roc_plotnode_x)
            roc_plot_y.append(roc_plotnode_y)
            if node.jus < 0:
                error_num += 1
    return roc_plot_x, roc_plot_y, error_num / len(test_list)


roc_plot_x, roc_plot_y, error_rate = roc(roc_list)
roc_plot_height_x, roc_plot_height_y, error_rate_height = roc(roc_list_height)
roc_plot_weight_x, roc_plot_weight_y, error_rate_weight = roc(roc_list_weight)

sample_height = np.linspace(150, 180, 100)
sample_weight = np.linspace(40, 80, 100)
# 将原始数据变成网格数据
Sample_height, Sample_weight = np.meshgrid(sample_height, sample_weight)
fig = plt.figure()
ax1 = fig.add_subplot(243)

ax3 = fig.add_subplot(242)
ax4 = fig.add_subplot(241)
ax7 = fig.add_subplot(244)
height_temp = f_singlefeature(sample_height, "height")
weight_temp = f_singlefeature(sample_weight, "weight")
ax3.scatter(sample_height, height_temp, marker="*", linewidths=0.3)
ax4.scatter(sample_weight, weight_temp, marker="*", linewidths=0.3)
for i in range(0, len(height_temp[0])):
    if abs(height_temp[0][i]) < 0.05:
        ax3.plot([sample_height[i], sample_height[i]], [-7, 7], linewidth=1, color='red')
        ax3.text(sample_height[i], 6, sample_height[i], fontdict={'color': 'r'})
ax3.scatter(man_height, np.zeros(len(man_height)), c='g', marker='*', linewidths=0.3)
ax3.scatter(woman_height, np.zeros(len(woman_height)), c='r', marker='*', linewidths=0.3)
for i in range(0, len(weight_temp[0])):
    if abs(weight_temp[0][i]) < 0.05:
        ax4.plot([sample_weight[i], sample_weight[i]], [-7, 7], linewidth=1, color='red')
        ax4.text(sample_weight[i], 6, sample_weight[i], fontdict={'color': 'r'})
ax4.scatter(man_weight, np.zeros(len(man_weight)), c='g', marker='*', linewidths=0.3)
ax4.scatter(woman_weight, np.zeros(len(woman_weight)), c='r', marker='*', linewidths=0.3)

# 填充颜色
ax1.contourf(Sample_height, Sample_weight, f(Sample_height, Sample_weight), 0, alpha=0)
# 绘制等高线,圈内为女生，圈外为男生
C = ax1.contour(Sample_height, Sample_weight, f(Sample_height, Sample_weight), 0, colors='black', linewidths=0.6)
# 显示各等高线的数据标签
ax1.clabel(C, inline=True, fontsize=10)
p1 = ax1.scatter(man_height, man_weight, c='g', marker='*', linewidths=0.4)
p2 = ax1.scatter(woman_height, woman_weight, c='r', marker='*', linewidths=0.4)
for h in test_list:
    if h.label == '男':
        ax1.scatter(h.height, h.weight, c='g', marker='*', linewidths=0.4)
    else:
        ax1.scatter(h.height, h.weight, c='r', marker='*', linewidths=0.4)
ax2 = fig.add_subplot(247)
ax5 = fig.add_subplot(245)
ax6 = fig.add_subplot(246)
ax2.plot(roc_plot_y, roc_plot_x, linewidth=1, color='red')
ax2.plot([0, 1], [0, 1], '--', color='blue')
ax5.plot(roc_plot_weight_y, roc_plot_weight_x, linewidth=1, color='red')
ax5.plot([0, 1], [0, 1], '--', color='blue')
ax6.plot(roc_plot_height_y, roc_plot_height_x, linewidth=1, color='red')
ax6.plot([0, 1], [0, 1], '--', color='blue')
ax7.scatter(sorted([h.height for h in test_list]), parzen_a)

label = ['boy', 'girl']
ax1.legend([p1, p2], label, loc=0)
ax3.legend([p1, p2], label, loc=0)
ax4.legend([p1, p2], label, loc=0)
ax1.set_xlabel('height/cm')
ax1.set_ylabel('weight/kg')
ax3.set_xlabel('height/cm')
ax3.set_ylabel('characteristic value')
ax4.set_xlabel('weight/kg')
ax4.set_ylabel('characteristic value')
ax2.set_xlabel('FPR')
ax2.set_ylabel('TPR')
ax5.set_xlabel('FPR')
ax5.set_ylabel('TPR')
ax6.set_xlabel('FPR')
ax6.set_ylabel('TPR')
print(error_rate)
print(error_rate_height)
print(error_rate_weight)
plt.show()
