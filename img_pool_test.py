#!/usr/bin/env python3
import numpy as np
import netCDF4 as nc
from PIL import Image
import matplotlib.pyplot as plt
from overlapping_pool import overlapping_pool


img = np.asarray(Image.open('img.jpg'))
whs = 64
pool_func = np.ptp

img_pool0 = overlapping_pool(img[:,:,0], whs, pool_func)
img_pool = np.empty((img_pool0.shape[0], img_pool0.shape[1], 3))
img_pool[:,:,0] = img_pool0; del(img_pool0)
img_pool[:,:,1] = overlapping_pool(img[:,:,1], whs, pool_func)
img_pool[:,:,2] = overlapping_pool(img[:,:,2], whs, pool_func)

# rescaling needed for RGB plotting
img_pool /= np.max(img_pool)

fig, axes = plt.subplots(2,1)
axes[0].imshow(img)
axes[1].imshow(img_pool)
plt.savefig('example/img_pool.png')
plt.show()
