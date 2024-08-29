import matplotlib.pyplot as plt

# 数据
x = [10, 20, 30, 40]
y1 = [19.87, 17.26, 12.89, 5.98]
y2 = [18.12, 13.65, 7.45, 4.87]
y3 = [22.89, 17.89, 14.12, 11.89]
y4 = [21.45,16.25,11.68,8.78]

# y3 = [14.32, 9.14, 5.36, 2.78]
# y4 = [13.7, 6.89, 3.89, 2.50]

# 设置字体大小
plt.rcParams.update({'font.size': 15})

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 绘制折线图，每种类型有两种颜色，并设置符号大小
plt.plot(x, y1, alpha=0.65,color='red', linestyle='-', marker='o', markeredgewidth=2,markersize=14, markeredgecolor='black',label='SST-2 - GBCL Prime')
plt.plot(x, y2, alpha=0.65,color='blue', linestyle='-', marker='o',markeredgewidth=2, markersize=14,markeredgecolor='black', label='SST-2 - GBCL Simplified')

plt.plot(x, y3, alpha=0.65,color='green', linestyle='-', marker='^', markeredgewidth=2,markersize=14,markeredgecolor='black', label='IMDB-2 - GBCL Prime')
plt.plot(x, y4, alpha=0.65,color='orange', linestyle='-', marker='^', markeredgewidth=2,markersize=14,markeredgecolor='black', label='IMDB-2 - GBCL Simplified')

# 添加标题和标签
plt.xlabel('Noise rate(%)')
plt.ylabel('Noise Correction(%)')

# 显示网格线
plt.grid(True)

# 显示图例
plt.legend()

# 导出高清图片
plt.savefig('2_class.png', dpi=600)

# 显示图形
plt.show()
