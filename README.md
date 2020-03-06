# overpool

Make a 2D pooling layer by walking in half windows over a 2D image (`whs`:= *window half-side*). Pooling is the operation of reducing image dimensions in a relevant way, usually for machine learning purposes. The pooling function is arbitrary (see in the "Examples" subsection bellow).


## Examples

### RGB image test

Using:
```python
whs = 64
pool_func = np.ptp  # (maximum value - minimum value)
```

![RGB image test](example/img_pool.png)


### NetCDF image test

Using:
```python
whs = 512
def pool_func(x):
    try:
        val = np.nanvar(x)
    except:              # if a window is full of NaNs,
        val = np.var(x)  # calculate simple variance
    return(val)
```

![NetCDF image test](example/netcdf_pool.png)
