#!/usr/bin/env python3


def fractal_dimension(Z, threshold=0.9):

    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])


    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def pool(img, whs=2, pool_func=np.std):
    # allocating
    ws = 2*whs
    rows, cols = img.shape
    assert not (ws>rows and ws>cols)
    pool_rows = rows//ws + (rows-whs)//ws
    pool_cols = cols//ws + (cols-whs)//ws
    extra_col, extra_row = 0, 0
    if rows%whs != 0: extra_row=1
    if cols%whs != 0: extra_col=1
    pooling_layer = np.empty((pool_rows+extra_row, pool_cols+extra_col))

    for xpi in range(pool_cols):
        xi = whs*xpi
        xf = xi + ws
        for ypi in range(pool_rows):
            yi = whs*ypi
            yf = yi + ws
            subimg = img[yi:yf,xi:xf]
            pooling_layer[ypi, xpi] = pool_func(subimg)
        if extra_row != 0:
            ypi = pool_rows
            subimg = img[-ws:,xi:xf]
            pooling_layer[ypi, xpi] = pool_func(subimg)

    if extra_col != 0:
        xpi = pool_cols
        for ypi in range(pool_cols):
            yi = whs*ypi
            yf = yi + ws
            subimg = img[yi:yf,-ws:]
            pooling_layer[ypi, xpi] = pool_func(subimg)
    return(pooling_layer)

img = np.asarray(Image.open('img.jpg'))
whs = 128
pool_func = fractal_dimension
img_pool0 = pool(img[:,:,0], whs, pool_func)
img_pool = np.empty((img_pool0.shape[0], img_pool0.shape[1], 3))
img_pool[:,:,0] = img_pool0
del img_pool0
img_pool[:,:,1] = pool(img[:,:,1], whs, pool_func)
img_pool[:,:,2] = pool(img[:,:,2], whs, pool_func)
img_pool /= np.max(img_pool)
#print(np.max(img_pool))
#print(img_pool)

fig, axes = plt.subplots(2,1)
axes[0].imshow(img)
axes[1].imshow(img_pool)
plt.show()
