# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 21:38:40 2025

@author: Eric
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
'''
mat = np.array([["1/9", "1/9", "1/9"],
                ["1/9", "1/9", "1/9"],
                ["1/9", "1/9", "1/9"]])

fig, ax = plt.subplots(figsize=(2, 2), dpi=100)

# 隐藏坐标轴
ax.axis('off')

# 绘制一个表格来模拟矩阵
table = plt.table(
    cellText=mat,
    loc='center',
    cellLoc='center'
)

# 调整字体和表格尺寸
table.set_fontsize(20)
table.scale(1, 4)  # 调高单元格高度

plt.savefig("matrix_temp.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.close(fig)

# 缩放到 200×200
img = Image.open("matrix_temp.png")
img_200 = img.resize((100, 100), Image.LANCZOS)
img_200.save("blur.png")'''

from PIL import Image, ImageDraw

# 图像大小
W, H = 100, 100

# 加号线宽
thickness = 10   # 可调粗细

# 创建白底图像
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

# 中心位置
cx, cy = W // 2, H // 2

# 画横线
draw.rectangle(
    (0, cy - thickness//2, W, cy + thickness//2),
    fill="black"
)

# 画竖线
draw.rectangle(
    (cx - thickness//2, 0, cx + thickness//2, H),
    fill="black"
)

img.save("plus_100x100.png")
print("✔ 已生成 plus_100x100.png")
