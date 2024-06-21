import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# 创建一个矩形来验证算法
def create_square(size, side_length):
    image = np.ones((size, size)) * 0.99
    start = (size - side_length) // 2
    end = start + side_length
    image[start:end, start:end] = 0.01
    return image

size = 100
side_length = 30
image = create_square(size, side_length)

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
plt.title('Original Square')
plt.subplot(1, 2, 2)
plt.imshow(segmentation, cmap='gray')
plt.title('Segmented Square')
plt.show()

# 保存图像
plt.imsave('square.png', image, cmap='gray')
plt.imsave('segmented_square.png', segmentation, cmap='gray')