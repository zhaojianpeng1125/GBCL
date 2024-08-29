import matplotlib.pyplot as plt

# 数据
x = [20, 40, 60, 80]
y1 = [27.89, 21.12, 13.32, 10.12]
y2 = [25.12,18.12,12.21,8.89]
y3 = [15.89,9.12,4.89,2.24]
y4 = [13.78,7.12,3.87,1.98]
y5 = [26.12,16.89,12.56,6.98]
y6 = [22.89,14.78,7.98,4.98]

# 设置字体大小
plt.rcParams.update({'font.size': 15})

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 绘制折线图，每种类型有两种颜色，并设置符号大小和边框
plt.plot(x, y1, alpha=0.65,color='red', linestyle='-', marker='o', markersize=14, markeredgecolor='black', markeredgewidth=2, label='Amazon-5 - GBCL Prime')
plt.plot(x, y2, alpha=0.65,color='blue', linestyle='-', marker='o', markersize=14, markeredgecolor='black', markeredgewidth=2, label='Amazon-5 - GBCL Simplified')

plt.plot(x, y3, alpha=0.65,color='green', linestyle='-', marker='^', markersize=14, markeredgecolor='black', markeredgewidth=2, label='SST-5 - GBCL Prime')
plt.plot(x, y4, alpha=0.65,color='orange', linestyle='-', marker='^', markersize=14, markeredgecolor='black', markeredgewidth=2, label='SST-5 - GBCL Simplified')

plt.plot(x, y5, alpha=0.65,color='pink', linestyle='-', marker='s', markersize=14, markeredgecolor='black', markeredgewidth=2, label='Yelp-5 - GBCL Prime')
plt.plot(x, y6, alpha=0.65,color='purple', linestyle='-', marker='s', markersize=14, markeredgecolor='black', markeredgewidth=2, label='Yelp-5 - GBCL Simplified')

# 添加标题和标签
plt.xlabel('Noise rate(%)')
plt.ylabel('Noise Correction(%)')

# 显示网格线
plt.grid(True)

# 显示图例
plt.legend()

# 导出高清图片
plt.savefig('5_class.png', dpi=600)

# 显示图形
plt.show()
