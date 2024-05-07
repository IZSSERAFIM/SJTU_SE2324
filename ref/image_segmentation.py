#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cvxpy as cp
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
size = 100  # Image size(size * size)
radius = 30  # Circle radius
image = create_image(size, radius)

plt.subplot(1, 2,  1)
plt.imshow(image, cmap='gray')
plt.title('Original image')
plt.show()

x = cp.Variable((size,size), boolean=True)
C1 = cp.multiply(x, image)
C2 = cp.multiply(1-x,1-image)
obj = cp.Minimize(cp.sum_squares(C1)+cp.sum_squares(C2))
problem = cp.Problem(obj)
problem.solve(solver=cp.ECOS_BB)
plt.subplot(1, 2,  2)
plt.title("x")
plt.imshow(x.value, cmap='gray')
plt.show()
