# overpool

Make a 2D pooling layer walking in half windows (`whs`:= window half-side) over a 2D image. Pooling is the operation of reducing image size in a relevant way, usually for machine learning purposes. The pooling function is arbitrary (see in the example bellow).


## Examples

Using:
```
whs = 64
pool_func = np.ptp  # (maximum value - minimum value)
```

![Image test](example/img_pool.png)


Using:
```
whs = 512
def pool_func(x):
    try:
        val = np.nanvar(x)
    except:              # if a window is full of NaNs,
        val = np.var(x)  # calculate simple variance
    return(val)
```

![Image test](example/netcdf_pool.png)
