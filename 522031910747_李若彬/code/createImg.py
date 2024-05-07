import numpy as np
import matplotlib.pyplot as plt

def create_image(size, radius):
    image = np.ones((size, size)) * 0.99
    center = size // 2
    Y, X = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
    mask = dist_from_center <= radius
    image[mask] = 0.01
    return image

size = 100
radius = 30
image = create_image(size, radius)

# 使用matplotlib保存图像
plt.imshow(image, cmap='gray')
plt.axis('off')  # 不显示坐标轴
plt.savefig('image.png', bbox_inches='tight', pad_inches = 0)