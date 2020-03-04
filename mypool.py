#!/usr/bin/env python3

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def pool(img, whs=2, pool_func=np.mean):
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


img = np.mean(np.asarray(Image.open('img.jpg')), axis=2)
#plt.imshow(img)
#plt.show()
img_pool = pool(img,64)
plot = plt.imshow(img_pool)
#plot.savefig('imgavg.jpg')
print(img.shape)
print(img_pool.shape)
