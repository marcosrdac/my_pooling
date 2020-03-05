#!/usr/bin/env python3
import numpy as np
import netCDF4 as nc
from PIL import Image
import matplotlib.pyplot as plt


def pool(img, whs=2, pool_func=np.std, pool_func_kwargs={}):
    # allocating
    ws = 2*whs
    rows, cols = img.shape
    assert not (ws>rows and ws>cols)
    pool_rows = rows//ws + (rows-whs)//ws
    pool_cols = cols//ws + (cols-whs)//ws
    extra_col, extra_row = 0, 0
    if rows%whs != 0 and (rows-whs)%whs != 0: extra_row=1
    if cols%whs != 0 and (cols-whs)%whs != 0: extra_col=1
    pooling_layer = np.empty((pool_rows+extra_row, pool_cols+extra_col))

    for xpi in range(pool_cols):
        xi = whs*xpi
        xf = xi + ws
        for ypi in range(pool_rows):
            yi = whs*ypi
            yf = yi + ws
            subimg = img[yi:yf,xi:xf]
            pooling_layer[ypi, xpi] = pool_func(subimg, **pool_func_kwargs)
        if extra_row != 0:
            ypi += 1
            subimg = img[-ws:,xi:xf]
            pooling_layer[ypi, xpi] = pool_func(subimg, **pool_func_kwargs)

    if extra_col != 0:
        xpi += 1
        for ypi in range(pool_rows):
            yi = whs*ypi
            yf = yi + ws
            subimg = img[yi:yf,-ws:]
            pooling_layer[ypi, xpi] = pool_func(subimg, **pool_func_kwargs)
    return(pooling_layer)


img_fp = '/home/marcosrdac/projects/oil_spill/netcdf/S1A_IW_SLC__1SDV_20181008T052755_20181008T052822_024040_02A081_3524_deb_Orb_ML_msk.nc'
img = nc.Dataset(img_fp)['Intensity_VV_db']
print(img.shape)

whs = 512
def pool_func(x):
    try:
        val = np.nanvar(x)
    except:
        val = np.var(x)
    return(val)
img_pool = pool(img, whs, pool_func)

fig, axes = plt.subplots(2,1)
axes[0].imshow(img)
axes[1].imshow(img_pool)
plt.show()
