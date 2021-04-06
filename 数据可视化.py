import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D

fontstyle = FontProperties(fname = "C:/Windows/Fonts/simsun.ttc",size=10)
df = pd.read_excel("D:/专业课/模式识别/forstudent/实验数据/genderdata/周五实验课数据统计_55.xlsx")
data = df.values
x1=[]
y1=[]
z1=[]
x2=[]
y2=[]
z2=[]
for row in data:
    if row[1]=="男":
        x1.append(row[2])
        y1.append(row[3])
        z1.append(row[4])
    else:
        x2.append(row[2])
        y2.append(row[3])
        z2.append(row[4])
fig = plt.figure()
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter3D(x1, y1, z1, cmap='Blues')
ax1.scatter3D(x2, y2, z2, cmap='Reds')
ax1.set_xlabel('身高',FontProperties=fontstyle)
ax1.set_ylabel('体重',FontProperties=fontstyle)
ax1.set_zlabel('鞋码',FontProperties=fontstyle)
ax2 = fig.add_subplot(222)
ax2.scatter(x1, y1, c='b', marker='o')
ax2.set_xlabel('身高', FontProperties=fontstyle)
ax2.set_ylabel('体重', FontProperties=fontstyle)
ax3 = fig.add_subplot(223)
ax3.scatter(y1, z1, c='r', marker='o')
ax3.set_xlabel('体重', FontProperties=fontstyle)
ax3.set_ylabel('鞋码', FontProperties=fontstyle)
ax4 = fig.add_subplot(224)
ax4.scatter(x1, z1, c='g', marker='o')
ax4.set_xlabel('身高', FontProperties=fontstyle)
ax4.set_ylabel('鞋码', FontProperties=fontstyle)
plt.show()

