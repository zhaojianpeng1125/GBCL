import matplotlib.pyplot as plt
import random

# 数据
x = [20, 40, 60, 80]  # x轴数据
y1 = [26, 18, 12, 8]   # 第一条折线的y轴数据
y2 = [28, 20, 13, 9]  # 第二条折线的y轴数据

# 对y值加上随机数
y1 = [y + random.uniform(-0.5, 0.5) for y in y1]
y2 = [y + random.uniform(-0.5, 0.5) for y in y2]

# 设置图表的大小和分辨率
plt.figure(figsize=(10, 6), dpi=300)

# 创建折线图
plt.plot(x, y1, label='GBCL Simplified', marker='o')  # 第一条折线
plt.plot(x, y2, label='GBCL Prime', marker='s')       # 第二条折线

# 添加标题和标签
plt.xlabel('Noise rate(%)', fontsize=18)  # 设置X轴标签字体大小
plt.ylabel('Noise Correction Rate(%)', fontsize=18)  # 设置Y轴标签字体大小

# 设置x轴刻度
plt.xticks(x)  # 仅显示x数据点

# 显示图例，设置位置在顶部水平展开，并调整字体大小
plt.legend(loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 1.15))

# 保存高质量的图表
plt.savefig('Amazon-5.png', dpi=600, bbox_inches='tight')

# 显示图表
plt.show()
