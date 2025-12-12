import numpy as np

x = np.random.random_sample((336,32,32))
window = 5

## method 1
cs = np.cumsum(x, axis=0)
ma1 = (cs[window:]-cs[:-window]) / window

## method 2
flt = np.ones(window) / window
ma2 = np.apply_along_axis(
        lambda m:np.convolve(m,flt,mode="valid"),
        axis=0,
        arr=x
        )

## method 3
ma3 = np.apply_along_axis(
        lambda m:np.convolve(m,flt,mode="valid"),
        axis=0,
        arr=x[1:]
        )

print(f"{ma1.shape=} {ma2.shape=} {ma3.shape=}")

assert np.all(np.isclose(ma1, ma2[1:]))
assert np.all(np.isclose(ma1, ma3))
