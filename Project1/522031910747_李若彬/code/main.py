import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# 从createImg.py中导入create_image函数并调用
from createImg import create_image

size = 100
radius = 30
image = create_image(size, radius)

# 创建一个cvxpy变量来表示分割矩阵
x = cp.Variable(image.shape, boolean=True)

# 定义目标函数
objective = cp.Minimize(cp.sum_squares(cp.multiply(x, image)) + cp.sum_squares(cp.multiply(1-x,1-image)))

# 定义约束条件
constraints = []

# 创建一个优化问题
problem = cp.Problem(objective, constraints)

# 求解这个问题
problem.solve(solver=cp.ECOS_BB)

# 获取最优的分割矩阵
segmentation = x.value

# 显示原始图像和分割后的图像并保存
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(segmentation, cmap='gray')
plt.title('Segmented Image')
plt.show()

# 保存图像
plt.imsave('image.png', image, cmap='gray')
plt.imsave('segmented_image.png', segmentation, cmap='gray')